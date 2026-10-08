import numpy as np
import base64
class DecodeError(RuntimeError):
    pass
def _unpack_10bit_blockwise(payload: bytes) -> np.ndarray:
    """
    Philips/SierraECG：每 5 個位元組（40 bits）→ 4 個 10-bit code（MSB-first）。
    c0 =  b0<<2 | b1>>6
    c1 = (b1&0x3F)<<4 | b2>>4
    c2 = (b2&0x0F)<<6 | b3>>2
    c3 = (b3&0x03)<<8 | b4
    尾端不足 5 bytes 的殘段丟棄。
    """
    n_full = (len(payload) // 5) * 5
    if n_full == 0:
        return np.array([], dtype=np.uint16)

    b = np.frombuffer(payload[:n_full], dtype=np.uint8).astype(np.uint16)
    out = np.empty((n_full // 5) * 4, dtype=np.uint16)

    b0, b1, b2, b3, b4 = b[0::5], b[1::5], b[2::5], b[3::5], b[4::5]
    out[0::4] = (b0 << 2) | (b1 >> 6)
    out[1::4] = ((b1 & 0x3F) << 4) | (b2 >> 4)
    out[2::4] = ((b2 & 0x0F) << 6) | (b3 >> 2)
    out[3::4] = ((b3 & 0x03) << 8) |  b4
    return out

# ----------------------------
# 差分還原（int16 逐步 two's-complement 環回）
# ----------------------------
def _delta_restore_int16(pre_delta_i16: np.ndarray, delta0: int, bias: int = 64) -> np.ndarray:
    """
    與 MATLAB 等價：每一步都在 int16 範圍內運算與環回（two's complement wrap）。
    z = 2*y - x - prev ; prev = pre[m] - bias
    """
    n = pre_delta_i16.size
    if n < 2:
        raise DecodeError(f"delta-restore needs >=2 samples, got {n}")

    sig = np.empty(n, dtype=np.int16)
    x = np.int16(pre_delta_i16[0])
    y = np.int16(pre_delta_i16[1])
    prev = np.int16(delta0)
    sig[0] = x
    sig[1] = y

    two = np.int16(2)
    b = np.int16(bias)

    old_err = np.seterr(over='ignore')  # MATLAB 自然 wrap，不視為錯誤
    for m in range(2, n):
        z = np.int16(np.int16(two * y - x) - prev)
        sig[m] = z
        prev = np.int16(pre_delta_i16[m] - b)
        x, y = y, z
    np.seterr(**old_err)
    return sig

# ----------------------------
# LZW 後 → 交錯成 int16 → 差分還原（order ∈ {'hilo','lohi'}）
# ----------------------------
def _pack_then_delta(de_u8: np.ndarray, delta0: int, *, order: str, bias: int) -> tuple[np.ndarray, np.ndarray]:
    """
    將 LZW 的 uint8 串：
      1) 若奇數，補 1 byte
      2) 'hilo'：前半 HI、後半 LO → 交錯 [HI,LO]
         'lohi'：前半 HI、後半 LO → 交錯 [LO,HI]
      3) 以小端 <i2 讀為 int16（delta 前）
      4) 差分還原（int16 wrap, 指定 bias）
    回傳： (signal_int16, pre_delta_int16)
    """
    L = int(de_u8.size)
    if (L % 2) != 0:
        de_u8 = np.pad(de_u8, (0, 1), mode='constant')
        L += 1

    half = L // 2
    hi, lo = de_u8[:half], de_u8[half:]

    inter = np.empty(L, dtype=np.uint8)
    if order == "hilo":  # [HI, LO]
        inter[0::2] = hi
        inter[1::2] = lo
    elif order == "lohi":  # [LO, HI]
        inter[0::2] = lo
        inter[1::2] = hi
    else:
        raise DecodeError(f"unknown pack order: {order}")

    pre_delta_i16 = np.frombuffer(inter.tobytes(), dtype='<i2')
    if pre_delta_i16.size < 2:
        raise DecodeError(f"too few int16 before delta: {pre_delta_i16.size}")

    sig = _delta_restore_int16(pre_delta_i16, delta0, bias=bias)
    return sig, pre_delta_i16
def _score_signal(sig: np.ndarray) -> tuple[float, float, float]:
    """
    回傳 (score, ptp_counts, mean_abs_diff)，分數越小越好。
    以 counts 的粗略合理範圍評分（等你有正確 uV/LSB 可再調）。
    """
    ptp = float(np.ptp(sig))
    diff = float(np.mean(np.abs(np.diff(sig.astype(np.int32))))) if sig.size > 1 else 0.0
    # 目標 counts 範圍（粗略）：300–6000
    lo_ok, hi_ok = 300.0, 6000.0
    if ptp < lo_ok:
        ptp_pen = (lo_ok - ptp) * 5.0
    elif ptp > hi_ok:
        ptp_pen = (ptp - hi_ok) * 0.5
    else:
        ptp_pen = 0.0
    score = ptp_pen + diff
    return score, ptp, diff

# ----------------------------
# 單一導程：尾端自適應 +（可選）pack/bias 自動選
# ----------------------------
def _decode_one_lead(
    payload: bytes,
    delta0: int,
    expect_n: int | None,
    t: list,
    *,
    auto_pack_bias: bool = True,
    strict_pack_order: str = "hilo",  # 'hilo' = [HI,LO]（MATLAB 等價）
    strict_bias: int = 64
) -> np.ndarray:
    """
    流程：
      payload → 10-bit（5→4, MSB-first）→ 檢查 → 嘗試 tail_drop ∈ {2,1,0}
      → LZW（經典）→ （auto）嘗試 [HI,LO]/[LO,HI] × bias {64,0}，擇優
                     （strict）固定使用 strict_pack_order + strict_bias
      → <i2 → 差分 → 選擇命中 expect_n 的結果；若只差 1，尾端自動補/砍。
    """
    # 10-bit
    codes_full = _unpack_10bit_blockwise(payload)
    if codes_full.size < 4:
        raise DecodeError("too few 10-bit codes")
    cmin, cmax = int(codes_full.min()), int(codes_full.max())
    if cmin < 0 or cmax > 1023:
        raise DecodeError(f"10-bit code out of range: min={cmin}, max={cmax}")
    t.append(("10bit_minmax", (cmin, cmax)))
    t.append(("10bit_head16", codes_full[:16].tolist()))

    best = None  # (err_to_expect, tail_drop, sig, meta_dict)

    for tail_drop in (2, 1, 0):
        codes = codes_full if tail_drop == 0 else codes_full[:-tail_drop]
        # LZW（經典，無 CLEAR/EOI）
        de_u8, _ = lzw2norm(codes.astype(np.uint16))
        if de_u8.size == 0 or de_u8.size > 16 * 1024 * 1024:
            t.append(("lzw_out_len_u8_bad", int(de_u8.size)))
            continue
        t.append(("lzw_out_len_u8_try", int(de_u8.size)))
        # ---- pack + delta ----
        candidates = []
        if auto_pack_bias:
            for order in ("hilo", "lohi"):
                for bias in (64, 0):
                    try:
                        sig_c, pre = _pack_then_delta(de_u8, delta0, order=order, bias=bias)
                    except DecodeError:
                        continue
                    score, ptp, mad = _score_signal(sig_c)
                    candidates.append((score, sig_c, order, bias, ptp, mad, pre))
        else:
            # MATLAB 嚴格模式
            sig_c, pre = _pack_then_delta(de_u8, delta0, order=strict_pack_order, bias=strict_bias)
            score, ptp, mad = _score_signal(sig_c)
            candidates = [(score, sig_c, strict_pack_order, strict_bias, ptp, mad, pre)]

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        score, sig_sel, order_sel, bias_sel, ptp_sel, mad_sel, pre_sel = candidates[0]

        n = int(sig_sel.size)
        err = 0 if expect_n is None else abs(n - expect_n)

        meta = {
            "tail_drop": tail_drop,
            "choose_pack_order": order_sel,
            "choose_bias": int(bias_sel),
            "delta_ptp_counts": float(ptp_sel),
            "delta_mean_abs_diff": float(mad_sel),
            "pack_head8_i16": pre_sel[:8].tolist(),
            "delta_head8": sig_sel[:8].tolist(),
            "delta_minmax": (int(sig_sel.min()), int(sig_sel.max())),
            "delta_restored_len": n,
        }

        # 命中期望長度 → 直接採用
        if expect_n is not None and n == expect_n:
            for k, v in meta.items():
                t.append((k, v))
            return sig_sel

        # 記錄最接近者
        if best is None or err < best[0]:
            best = (err, tail_drop, sig_sel, meta)

    if best is None:
        raise DecodeError("no viable decoding candidate for this lead")

    # 沒有完美命中，採用最接近者；若只差 1，自動補/砍
    err, tail_drop, sig_sel, meta = best
    for k, v in meta.items():
        t.append((k, v))
    if expect_n is not None and err == 1:
        if sig_sel.size < expect_n:
            sig_sel = np.concatenate([sig_sel, sig_sel[-1:]])
            t.append(("pad_tail_one", True))
        else:
            sig_sel = sig_sel[:-1]
            t.append(("trim_tail_one", True))
        t.append(("delta_restored_len_after_fix", int(sig_sel.size)))

    return sig_sel

# ----------------------------
# 主解碼：Base64 → 12導程還原（與 MATLAB 對齊 + 強韌化）
# ----------------------------
def decode_philips_waveform(
    base64_str: str,
    *,
    n_leads: int = 12,
    expected_samples_per_lead: int | None = 5500,  # 500 Hz × 11 s
    verbose: bool = True,
    auto_pack_bias: bool = True,        # ← 預設自動挑 pack+bias；要嚴格對 MATLAB 則設 False
    strict_pack_order: str = "hilo",    # 嚴格模式時固定為 [HI,LO]（MATLAB 等價）
    strict_bias: int = 64               # 嚴格模式時的 bias
):
    try:
        decoded = base64.b64decode(base64_str, validate=False)
    except Exception as e:
        raise DecodeError(f"Base64 decode failed: {e}") from e

    total_len = len(decoded)
    if total_len == 0:
        raise DecodeError("empty base64 payload")

    lead_signals: dict[str, np.ndarray] = {}
    trace_per_lead: dict[str, list] = {}

    offset = 0
    for li in range(n_leads):
        name = f"Lead_{li+1}"
        t: list = []
        trace_per_lead[name] = t

        # 8-byte header（LE）
        if offset + 8 > total_len:
            raise DecodeError(f"[{name}] insufficient header at {offset}/{total_len}")

        h = decoded[offset: offset+8]
        datasize = int.from_bytes(h[0:4], 'little', signed=False)
        codeone  = int.from_bytes(h[4:6], 'little', signed=False)  # 觀察用
        delta0   = int.from_bytes(h[6:8], 'little', signed=True)
        t.append(("header_datasize", datasize))
        t.append(("header_codeone",  codeone))
        t.append(("header_delta0",   delta0))

        start = offset + 8
        end   = start + datasize
        if end > total_len:
            raise DecodeError(f"[{name}] payload out of range: {start}->{end}/{total_len}")
        if datasize == 0 or datasize > 4 * 1024 * 1024:  # 超寬鬆上限
            raise DecodeError(f"[{name}] suspicious datasize={datasize}")

        payload = decoded[start:end]
        offset = end

        # 單導解碼（尾端自適應 + pack/bias 自動/嚴格）
        sig = _decode_one_lead(
            payload, delta0, expected_samples_per_lead, t,
            auto_pack_bias=auto_pack_bias,
            strict_pack_order=strict_pack_order,
            strict_bias=strict_bias
        )

        # 若仍與期望差距 > 1，嚴格提示
        if expected_samples_per_lead is not None and abs(sig.size - expected_samples_per_lead) > 1:
            raise DecodeError(f"[{name}] sample count mismatch: got={sig.size}, "
                              f"expected={expected_samples_per_lead}")

        lead_signals[name] = sig

    # 有些檔在 parsedwaveforms 後還有其他 bytes（如 checksum/bank）；不影響 12 導
    if offset != total_len and verbose:
        print(f"[WARN] payload not fully consumed: consumed={offset}, total={total_len}")

    profile = {
        "trace_per_lead": trace_per_lead,
        "consumed_bytes": offset,
        "total_bytes": total_len,
    }

    if verbose:
        print(f"[OK] decoded_len={total_len}, consumed={offset}")
        for k in range(n_leads):
            nm = f"Lead_{k+1}"
            tr = trace_per_lead.get(nm, [])
            def _get(tag, default=None):
                val = default
                for (kk, vv) in tr:
                    if kk == tag:
                        val = vv
                return val
            ds  = _get("header_datasize")
            d0  = _get("header_delta0")
            td  = _get("tail_drop", _get("choose_tail_drop", _get("choose_tail_drop_near")))
            rl  = _get("delta_restored_len_after_fix", _get("delta_restored_len"))
            po  = _get("choose_pack_order")
            bs  = _get("choose_bias")
            print(f" - {nm}: datasize={ds}, delta0={d0}, tail_drop={td}, pack={po}, bias={bs}, restored={rl}")

    return lead_signals, profile
class LZWError(RuntimeError):
    pass

def lzw2norm(codes: np.ndarray):
    """
    Classic LZW (no CLEAR/EOI), 10-bit code space (0..1023).
    行為對齊 MATLAB lzw2norm.m：
      - 初始字典 0..255 對應單一位元組
      - next_code 從 256 起
      - 支援 KwKwK（k == next_code -> s + s[:1]）
    """
    if not isinstance(codes, np.ndarray) or codes.dtype != np.uint16:
        raise LZWError("lzw2norm: Input must be np.ndarray of dtype uint16")
    if codes.size == 0:
        return np.frombuffer(b"", dtype=np.uint8), {}

    cmin = int(codes.min()); cmax = int(codes.max())
    if cmin < 0 or cmax > 1023:
        raise LZWError(f"lzw2norm: code out of 10-bit range (min={cmin}, max={cmax})")

    # 初始字典：單一位元組
    table = {i: bytes([i]) for i in range(256)}
    next_code = 256

    it = iter(int(x) for x in codes)
    try:
        first = next(it)
    except StopIteration:
        return np.frombuffer(b"", dtype=np.uint8), table

    if not (0 <= first <= 255):
        raise LZWError(f"lzw2norm: Bad start code {first} (expected 0..255)")

    s = table[first]
    out = bytearray(s)

    # 寬鬆防爆：單導未壓縮照理 ~10–20KB；這裡給 8MB 上限避免錯位時暴增
    SOFT_OUT_LIMIT = 8 * 1024 * 1024

    for k in it:
        if k in table:
            entry = table[k]
        elif k == next_code:
            entry = s + s[:1]     # KwKwK
        else:
            raise LZWError(f"lzw2norm: Bad compressed element {k} (next_code={next_code})")

        out += entry
        if len(out) > SOFT_OUT_LIMIT:
            raise LZWError(f"lzw2norm: output too large ({len(out)} bytes) — wrong framing?")

        table[next_code] = s + entry[:1]
        next_code += 1
        s = entry

    return np.frombuffer(out, dtype=np.uint8), table
FALLBACK_SAMPLE_RATE = 500.0
FALLBACK_UV_PER_LSB  = 5.0
MAP_B = {
    "Lead_1":"I","Lead_2":"II","Lead_3":"III",
    "Lead_4":"aVR","Lead_5":"aVL","Lead_6":"aVF",
    "Lead_7":"V1","Lead_8":"V2","Lead_9":"V3",
    "Lead_10":"V4","Lead_11":"V5","Lead_12":"V6",
}
def counts_to_mV(raw_counts, uV_per_lsb=None, offset_counts=None,
                 fallback_uV_per_lsb=5.0, use_median_if_no_offset=True):
    """ADC counts → mV"""
    x = np.asarray(raw_counts, dtype=np.float32)
    if uV_per_lsb is None:
        uV_per_lsb = float(fallback_uV_per_lsb)
    if offset_counts is None:
        offset_counts = float(np.median(x)) if use_median_if_no_offset else 0.0
    return (x - offset_counts) * (uV_per_lsb / 1000.0)
def attach_mV(lead_data: dict):
    warned = False
    for info in lead_data.values():
        if info.get("uV_per_lsb") is None and not warned:
            print(f"[WARN] XML 無 uV/LSB，改用備援 {FALLBACK_UV_PER_LSB} µV/LSB")
            warned = True
        mv = counts_to_mV(
            info["signal"],
            uV_per_lsb=info.get("uV_per_lsb"),
            offset_counts=info.get("offset_counts"),
            fallback_uV_per_lsb=FALLBACK_UV_PER_LSB,
        )
        info["signal_mV"] = np.asarray(mv, float)

def alias_standard_names_bankB(lead_data: dict):
    for src, std in MAP_B.items():
        if src in lead_data and std not in lead_data:
            lead_data[std] = {
                "signal_mV": np.asarray(lead_data[src]["signal_mV"], float),
                "sample_rate": float(lead_data[src].get("sample_rate", FALLBACK_SAMPLE_RATE)),
            }

def force_chest_bank_B(lead_data: dict):
    for i in range(1, 7):
        src = f"Lead_{i+6}"
        dst = f"V{i}"
        if src in lead_data and "signal_mV" in lead_data[src]:
            s  = np.asarray(lead_data[src]["signal_mV"], float)
            fs = float(lead_data[src].get("sample_rate", FALLBACK_SAMPLE_RATE))
            lead_data[dst] = {"signal_mV": s.copy(), "sample_rate": fs}
    print("[MAP] chest bank forced: B (V1..V6 ← Lead_7..12)")

def derive_limb_from_I_II(lead_data: dict):
    if "I" not in lead_data or "II" not in lead_data:
        return
    I  = np.asarray(lead_data["I"]["signal_mV"],  float)
    II = np.asarray(lead_data["II"]["signal_mV"], float)
    N  = min(I.size, II.size); I, II = I[:N], II[:N]
    fs = float(lead_data["I"].get("sample_rate", FALLBACK_SAMPLE_RATE))
    combos = {
        "III": II - I,
        "aVR": -(I + II)/2.0,
        "aVL":  I - II/2.0,
        "aVF":  II - I/2.0,
    }
    for k, y in combos.items():
        lead_data[k] = {"signal_mV": np.asarray(y, float), "sample_rate": fs}

# -----------------------------------------------------------------------------
# CardioSketch input adapter
# -----------------------------------------------------------------------------
import xml.etree.ElementTree as ET


def _first_element_by_local_name(root, local_name: str):
    """Find the first XML element whose local tag name matches (namespace-safe)."""
    wanted = local_name.lower()
    for elem in root.iter():
        tag = elem.tag
        if isinstance(tag, str) and tag.rsplit("}", 1)[-1].lower() == wanted:
            return elem
    return None


def _find_sample_rate(root, fallback: float = FALLBACK_SAMPLE_RATE) -> float:
    """Read a sampling-rate field if present; otherwise use the validated 500-Hz fallback."""
    names = {
        "samplerate", "samplingrate", "samplespersecond",
        "samplefrequency", "frequency", "fs"
    }
    for elem in root.iter():
        tag = elem.tag
        if not isinstance(tag, str):
            continue
        local = tag.rsplit("}", 1)[-1].lower()
        if local in names and elem.text and elem.text.strip():
            try:
                fs = float(elem.text.strip())
                if fs > 0:
                    return fs
            except ValueError:
                pass
        for key, value in elem.attrib.items():
            if key.rsplit("}", 1)[-1].lower() in names:
                try:
                    fs = float(value)
                    if fs > 0:
                        return fs
                except (TypeError, ValueError):
                    pass
    return float(fallback)


def _find_numeric_metadata(root, names):
    """Find the first numeric attribute/element matching one of names."""
    wanted = {n.lower() for n in names}
    for elem in root.iter():
        for key, value in elem.attrib.items():
            if key.rsplit("}", 1)[-1].lower() in wanted:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        tag = elem.tag
        if isinstance(tag, str) and tag.rsplit("}", 1)[-1].lower() in wanted:
            if elem.text and elem.text.strip():
                try:
                    return float(elem.text.strip())
                except ValueError:
                    pass
    return None


def load_chung_long_xml(path, *, verbose=True):
    """Load the validated Zhongrong/Philips Sierra XML format for CardioSketch.

    Returns
    -------
    sig : np.ndarray
        ECG matrix with shape (n_samples, 12), in mV.
    fs : float
        Sampling frequency in Hz.
    leads : list[str]
        Standard 12-lead names in CardioSketch order.

    The source-specific work is deliberately kept here: XML parsing, Base64 /
    10-bit / LZW / delta decoding, ADC-count-to-mV conversion, lead naming,
    and the validated I/II-derived limb leads. The downstream CardioSketch
    detection pipeline remains source-agnostic.
    """
    path = str(path)
    root = ET.parse(path).getroot()

    waveform_elem = _first_element_by_local_name(root, "parsedwaveforms")
    if waveform_elem is None:
        raise ValueError("Zhongrong XML does not contain <parsedwaveforms>.")

    waveform_base64 = (waveform_elem.text or "").strip()
    if not waveform_base64:
        raise ValueError("<parsedwaveforms> is empty.")

    duration_ms = None
    for key, value in waveform_elem.attrib.items():
        if key.rsplit("}", 1)[-1].lower() == "durationperchannel":
            try:
                duration_ms = float(value)
            except ValueError:
                pass
            break

    # The validated files use 11 s / 5500 samples = 500 Hz. If the XML gives
    # an explicit sampling rate, prefer it; otherwise preserve that fallback.
    fs = _find_sample_rate(root)
    if verbose:
        print(f"[XML] loading: {path}")
        print(f"[XML] sample rate: {fs:g} Hz")
        if duration_ms is not None:
            print(f"[XML] duration/channel: {duration_ms:g} ms")

    expected_samples = None
    if duration_ms is not None and fs > 0:
        expected_samples = int(round(duration_ms / 1000.0 * fs))
    if expected_samples is None:
        expected_samples = 5500

    lead_signals, profile = decode_philips_waveform(
        waveform_base64,
        n_leads=12,
        expected_samples_per_lead=expected_samples,
        verbose=verbose,
        auto_pack_bias=True,
        strict_pack_order="hilo",
        strict_bias=64,
    )

    # Build the same lead_data structure used by the validated old workflow.
    uV_per_lsb = _find_numeric_metadata(
        root,
        {"signalresolution", "microvoltsperbit", "amplitudeunitsperbit", "gainuvperbit"},
    )
    offset_counts = _find_numeric_metadata(
        root,
        {"signaloffset", "baseline", "baselineoffset", "digitalbaseline"},
    )

    lead_data = {}
    for name, signal in lead_signals.items():
        lead_data[name] = {
            "signal": np.asarray(signal),
            "sample_rate": fs,
            "uV_per_lsb": uV_per_lsb,
            "offset_counts": offset_counts,
        }

    attach_mV(lead_data)
    alias_standard_names_bankB(lead_data)
    force_chest_bank_B(lead_data)
    # Validated reconstruction for this XML source.
    derive_limb_from_I_II(lead_data)

    leads = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

    missing = [lead for lead in leads if lead not in lead_data]
    if missing:
        raise ValueError(f"Decoded ECG is missing leads: {missing}")

    n = min(np.asarray(lead_data[lead]["signal_mV"]).size for lead in leads)
    sig = np.column_stack([
        np.asarray(lead_data[lead]["signal_mV"], dtype=float)[:n]
        for lead in leads
    ])

    if verbose:
        print(f"[XML] decoded matrix: {sig.shape} (samples, leads)")
        print(f"[XML] leads: {', '.join(leads)}")

    return sig, float(fs), leads

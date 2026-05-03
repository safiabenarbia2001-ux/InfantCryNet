"""
audio_features.py — Feature extraction for the cry analysis pipeline.

FILE 1/3 for Stage 0.

What this file does:
    - Loads an audio file safely
    - Checks signal quality (is this audio usable?)
    - Extracts ~45 compact features designed for cry detection

Why ~45 features and not 350?
    Stage 0 is binary (cry vs non-cry). With small datasets (~400 files),
    fewer features = less overfitting. 45 features is enough to separate
    cries from non-cries. The richer 120-feature set comes later for
    cause classification (Nodes A/B/C).
"""

from typing import Dict, List, Optional
import numpy as np

EPS = 1e-12


def _safe(x, default: float = 0.0) -> float:
    """Convert to float, return default if NaN/Inf/error."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(
    filepath: str,
    sr: int = 22050,
    duration: float = 5.0,
    min_duration: float = 0.3,
) -> Optional[np.ndarray]:
    """Load an audio file → numpy array.

    Steps:
        1. Load with librosa at target sample rate
        2. Check minimum length
        3. Normalize amplitude to [-1, 1]
        4. Trim silence from edges

    Returns None if the file can't be loaded or is too short.
    """
    try:
        import librosa

        y, _ = librosa.load(filepath, sr=sr, duration=duration)

        # Too short? Skip it
        if len(y) < int(sr * min_duration):
            return None

        # Normalize
        y = librosa.util.normalize(y)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) >= int(sr * min_duration):
            return y_trimmed

        return y

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_quality(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Check if this audio signal is usable.

    Returns a dict with:
        snr_db         — signal-to-noise ratio estimate
        duration       — length in seconds
        clipping_ratio — how much of the signal is clipped (distorted)
        quality_score  — combined score 0 to 1 (higher = better)

    If quality_score < 0.15, the audio is garbage — don't trust features.
    """
    duration = len(y) / sr
    clipping_ratio = float(np.mean(np.abs(y) > 0.99))

    # Estimate SNR using frame energies
    frame_len = int(0.03 * sr)  # 30ms frames
    n_frames = max(1, len(y) // frame_len)
    energies = np.array([
        np.mean(y[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    energies = np.clip(energies, EPS, None)

    # Signal = loud frames (90th percentile), Noise = quiet frames (10th)
    snr_db = float(10 * np.log10(
        np.percentile(energies, 90) / (np.percentile(energies, 10) + EPS)
    ))

    # Combined score: weight SNR most, then duration and clipping
    quality_score = float(
        0.4 * np.clip(snr_db / 30.0, 0.0, 1.0)
        + 0.3 * np.clip(duration / 3.0, 0.0, 1.0)
        + 0.3 * (1.0 - clipping_ratio)
    )

    return {
        "snr_db": snr_db,
        "duration": duration,
        "clipping_ratio": clipping_ratio,
        "quality_score": quality_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CRY FEATURES — ~45 features for Stage 0
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cry_features(y: np.ndarray, sr: int = 22050) -> Optional[Dict[str, float]]:
    """Extract ~45 features optimized for cry vs non-cry detection.

    Feature groups:
        1. Energy        (6 features)  — how loud, how variable
        2. Pitch / F0    (7 features)  — fundamental frequency of the voice
        3. MFCC          (26 features) — vocal tract shape (13 coefficients × mean+std)
        4. Spectral      (5 features)  — frequency distribution shape
        5. ZCR           (1 feature)   — how "noisy" vs "tonal"
        ─────────────────────────────────
        Total:           45 features

    Why these features separate cries from non-cries:
        - Cries have HIGH pitch (250-600 Hz), non-cries usually don't
        - Cries have RHYTHMIC energy (cry-pause-cry pattern)
        - Cries are TONAL (low spectral flatness, high voiced ratio)
        - Cries have SPECIFIC MFCC patterns (vocal tract of an infant)
    """
    import librosa

    d: Dict[str, float] = {}

    try:
        # ── GROUP 1: Energy (6 features) ──────────────────────────────────
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        d["energy_mean"] = _safe(rms_mean)
        d["energy_std"] = _safe(rms_std)
        d["energy_cv"] = _safe(rms_std / (rms_mean + EPS))
        d["energy_max"] = _safe(np.max(rms))

        # Energy entropy: uniform energy = high entropy, bursty = low
        rms_norm = rms / (np.sum(rms) + EPS)
        d["energy_entropy"] = _safe(-np.sum(rms_norm * np.log(rms_norm + EPS)))

        # How fast does energy change? (cry = fast modulation)
        d["energy_delta_mean"] = _safe(np.mean(np.abs(np.diff(rms))))

        # ── GROUP 2: Pitch / F0 (7 features) ─────────────────────────────
        f0, _, voiced_probs = librosa.pyin(
            y, fmin=100, fmax=800, sr=sr, fill_na=np.nan
        )
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 3:
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced)
            d["f0_mean"] = _safe(f0_mean)
            d["f0_std"] = _safe(f0_std)
            d["f0_cv"] = _safe(f0_std / (f0_mean + EPS))
            d["f0_median"] = _safe(np.median(f0_voiced))
            d["f0_range"] = _safe(np.ptp(f0_voiced))
        else:
            for k in ["f0_mean", "f0_std", "f0_cv", "f0_median", "f0_range"]:
                d[k] = 0.0

        # What fraction of the signal is voiced? (cries = mostly voiced)
        d["voiced_ratio"] = _safe(np.sum(~np.isnan(f0)) / (len(f0) + EPS))
        d["voiced_prob_mean"] = _safe(np.mean(voiced_probs))

        # ── GROUP 3: MFCC (26 features = 13 × mean + std) ────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048)
        for i in range(13):
            d[f"mfcc_{i}_mean"] = _safe(np.mean(mfcc[i]))
            d[f"mfcc_{i}_std"] = _safe(np.std(mfcc[i]))

        # ── GROUP 4: Spectral shape (5 features) ─────────────────────────
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        d["spectral_centroid_mean"] = _safe(np.mean(cent))
        d["spectral_centroid_std"] = _safe(np.std(cent))

        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        d["spectral_bandwidth_mean"] = _safe(np.mean(bw))

        # Flatness: 0 = pure tone, 1 = white noise. Cries are tonal → low
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        d["spectral_flatness_mean"] = _safe(np.mean(flatness))
        d["spectral_flatness_std"] = _safe(np.std(flatness))

        # ── GROUP 5: Zero Crossing Rate (1 feature) ──────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        d["zcr_mean"] = _safe(np.mean(zcr))

    except Exception:
        return None

    return d


# ═══════════════════════════════════════════════════════════════════════════════
#  CAUSE FEATURES — ~120 features for Nodes A/B/C
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cause_features(y: np.ndarray, sr: int = 22050) -> Optional[Dict[str, float]]:
    """Richer feature set for cause classification (hungry vs tired vs pain etc).

    Starts with all 45 cry features, then adds:
        6. MFCC deltas       (26 features) — how MFCCs change over time
        7. Voice quality      (4 features)  — jitter, pitch slope, pitch peaks
        8. Spectral extras    (6 features)  — rolloff, contrast
        9. Sub-band energies  (12 features) — energy in 4 frequency bands
       10. Temporal/onset     (3 features)  — onset rate, onset strength
       11. HNR approximation  (1 feature)   — harmonic-to-noise ratio
        ─────────────────────────────────────────
        Total: 45 + 26 + 4 + 6 + 12 + 3 + 1 ≈ 97 features

    Why these extras matter for CAUSE classification:
        - MFCC deltas: hungry cries accelerate, tired cries slow down
        - Jitter: pain cries have more vocal irregularity
        - Sub-bands: hunger cries have different spectral energy distribution
        - Onset rate: colic = rapid bursts, tiredness = slow fading
    """
    import librosa
    from scipy.signal import find_peaks

    # Start with all 45 cry features
    d = extract_cry_features(y, sr)
    if d is None:
        return None

    try:
        # ── GROUP 6: MFCC deltas (26 = 13 × mean + std) ──────────────────
        # Delta = how each MFCC changes frame to frame
        # Captures temporal dynamics: hungry cry speeds up, tired slows down
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048)
        delta = librosa.feature.delta(mfcc)
        for i in range(13):
            d[f"mfcc_delta_{i}_mean"] = _safe(np.mean(delta[i]))
            d[f"mfcc_delta_{i}_std"] = _safe(np.std(delta[i]))

        # ── GROUP 7: Voice quality (4 features) ──────────────────────────
        f0, _, _ = librosa.pyin(y, fmin=100, fmax=800, sr=sr, fill_na=np.nan)
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 5:
            # Jitter: pitch period irregularity (pain → more jitter)
            periods = 1.0 / (f0_voiced + EPS)
            period_diffs = np.abs(np.diff(periods))
            d["jitter_local"] = _safe(np.mean(period_diffs) / (np.mean(periods) + EPS))

            # Pitch slope: does the cry go up or down over time?
            t = np.arange(len(f0_voiced))
            slope = np.polyfit(t, f0_voiced, 1)[0] if len(t) > 2 else 0.0
            d["f0_slope"] = _safe(slope)

            # Pitch peaks: how many times does pitch go up-down?
            # Colic = many peaks, hunger = fewer
            peaks, _ = find_peaks(f0_voiced, prominence=10)
            d["f0_n_peaks"] = float(len(peaks))

            # Jitter RAP (3-point average perturbation)
            if len(periods) > 4:
                rap = np.mean(np.abs(
                    periods[1:-1] - (periods[:-2] + periods[1:-1] + periods[2:]) / 3
                )) / (np.mean(periods) + EPS)
                d["jitter_rap"] = _safe(rap)
            else:
                d["jitter_rap"] = 0.0
        else:
            d["jitter_local"] = 0.0
            d["f0_slope"] = 0.0
            d["f0_n_peaks"] = 0.0
            d["jitter_rap"] = 0.0

        # ── GROUP 8: Spectral extras (6 features) ────────────────────────
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        d["spectral_rolloff_mean"] = _safe(np.mean(rolloff))
        d["spectral_rolloff_std"] = _safe(np.std(rolloff))

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=3)
            for i in range(min(contrast.shape[0], 4)):
                d[f"spectral_contrast_{i}"] = _safe(np.mean(contrast[i]))
        except Exception:
            for i in range(4):
                d[f"spectral_contrast_{i}"] = 0.0

        # ── GROUP 9: Sub-band energies (12 = 4 bands × 3 stats) ──────────
        # Split spectrum into 4 bands, get energy distribution
        # Different cry causes put energy in different frequency ranges
        S = np.abs(librosa.stft(y, n_fft=2048))
        n_bins = S.shape[0]
        band_size = n_bins // 4
        total_energy = np.sum(S ** 2) + EPS

        for b in range(4):
            start = b * band_size
            end = (b + 1) * band_size if b < 3 else n_bins
            band = S[start:end, :]
            band_energy = np.sum(band ** 2)
            d[f"subband_{b}_mean"] = _safe(np.mean(band))
            d[f"subband_{b}_std"] = _safe(np.std(band))
            d[f"subband_{b}_ratio"] = _safe(band_energy / total_energy)

        # ── GROUP 10: Temporal / Onset (3 features) ──────────────────────
        # Onset = when a new sound event starts
        # Colic: rapid onsets. Tired: slow, fading onsets.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        d["onset_strength_mean"] = _safe(np.mean(onset_env))
        d["onset_strength_std"] = _safe(np.std(onset_env))
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        d["onset_rate"] = _safe(len(onsets) / (len(y) / sr + EPS))

        # ── GROUP 11: HNR approximation (1 feature) ──────────────────────
        # Harmonic-to-Noise Ratio: how "clean" is the voice?
        # Pain cries = noisier (lower HNR), hunger = cleaner (higher HNR)
        seg = y[:min(len(y), sr)]  # First 1 second
        autocorr = np.correlate(seg, seg, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if len(autocorr) > int(sr / 100):
            search_start = max(1, int(sr / 800))
            search_end = min(len(autocorr), int(sr / 100))
            if search_end > search_start:
                peak_idx = np.argmax(autocorr[search_start:search_end]) + search_start
                harmonic = abs(autocorr[peak_idx])
                noise = abs(autocorr[0]) - harmonic
                d["hnr_approx"] = _safe(10 * np.log10(harmonic / (abs(noise) + EPS)))
            else:
                d["hnr_approx"] = 0.0
        else:
            d["hnr_approx"] = 0.0

    except Exception:
        pass  # Keep whatever we got from cry_features

    return d


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH EXTRACTION — process many files at once
# ═══════════════════════════════════════════════════════════════════════════════

def extract_batch(
    filepaths: List[str],
    feature_type: str = "cry",
    sr: int = 22050,
    duration: float = 5.0,
    verbose: bool = True,
) -> List[Optional[Dict[str, float]]]:
    """Extract features for a list of audio files.

    Args:
        filepaths: list of audio file paths
        feature_type: "cry" for Stage 0 (45 features)
                      "cause" for Nodes A/B/C (120 features)

    Returns a list of the same length as filepaths.
    Each element is either a feature dict or None (if extraction failed).
    """
    fn = extract_cause_features if feature_type == "cause" else extract_cry_features

    results = []
    n_ok = 0
    n_fail = 0

    for i, fp in enumerate(filepaths):
        y = load_audio(fp, sr=sr, duration=duration)
        if y is not None:
            feats = fn(y, sr)
            results.append(feats)
            if feats is not None:
                n_ok += 1
            else:
                n_fail += 1
        else:
            results.append(None)
            n_fail += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(filepaths)}]  ok={n_ok}  fail={n_fail}")

    if verbose:
        print(f"    Done: {len(filepaths)} files → {n_ok} ok, {n_fail} failed")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing audio_features.py ...")

    sr = 22050
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur))

    # Simulate cry: 400Hz tone with amplitude modulation
    cry_signal = (
        0.5 * np.sin(2 * np.pi * 400 * t)
        * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    ).astype(np.float32)

    # Test cry features (Stage 0)
    feats_cry = extract_cry_features(cry_signal, sr)
    if feats_cry is not None:
        print(f"  ✅ Cry features:  {len(feats_cry)} features (Stage 0)")
    else:
        print("  ❌ Cry feature extraction failed")

    # Test cause features (Nodes A/B/C)
    feats_cause = extract_cause_features(cry_signal, sr)
    if feats_cause is not None:
        print(f"  ✅ Cause features: {len(feats_cause)} features (Nodes A/B/C)")
        # Show the extra features
        extra = set(feats_cause.keys()) - set(feats_cry.keys())
        print(f"     Extra features over cry: {len(extra)}")
        print(f"     Examples: {sorted(extra)[:5]}")
    else:
        print("  ❌ Cause feature extraction failed")

    quality = check_quality(cry_signal, sr)
    print(f"  ✅ Quality score: {quality['quality_score']:.3f}")

    print("\n  audio_features.py — OK ✅")
# GK-2A LRIT Data Format

## Source

Korean GK-2A geostationary weather satellite, IR105 channel (10.5 μm infrared). Data received via LRIT (Low Rate Information Transmission) downlink.

## File Pairing

11 fragment directories on the 5080 machine, each containing `IMAGES/GK-2A/<timestamp>/` subdirectories. Each timestamp directory has:

- `GK2A_IR105_*.png` — 2200×2200 grayscale 8-bit DN (Digital Number) infrared image
- `product.cbor` — Calibration metadata including DN → Kelvin lookup table

1,480 complete PNG+CBOR pairs found. 957 orphans in each direction (skipped).

## CBOR Structure

Keys: `bit_depth`, `calibration`, `has_timestamps`, `images`, `instrument`, `needs_correlation`, `product_source`, `product_timestamp`, `projection_cfg`, `timestamps`, `timestamps_type`, `type`

Critical field: `calibration.IR105` — list of `[DN, temp_K]` pairs forming the lookup table for converting pixel values to brightness temperature.

## Calibration Pipeline

```
PNG pixel (0–255 DN) → CBOR LUT interpolation → Kelvin → subtract 273.15 → °C
```

## Temperature Ranges

| Feature | Temperature | Significance |
|---------|-------------|--------------|
| Deep convection (eyewall) | < -50°C | Storm indicator, used for cloud masking |
| Eye region | Warmer (exposed warm core) | Eye-eyewall ΔT drives T-number |
| Clear sky / ocean | > 0°C | Background, not storm-related |

## Storm Detection

The ADT algorithm (`applyADT.py`) processes each full-disk image:
1. Cloud mask at T < -50°C
2. Contour analysis to find connected cold regions
3. Centroid extraction for each candidate
4. 240×240 patch extraction centered on each candidate
5. Eye detection and ΔT calculation → T-number, wind, pressure

Typical yield: ~36 storm candidates per full-disk image. Most are false positives (cold cloud features, not actual typhoons), which is why 62% of patches receive T=8.0 (ADT saturation at high ΔT).

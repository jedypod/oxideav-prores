//! The quantised-coefficient hook (`encoder::quantized`).
//!
//! `encode_quantized_frame(quantize_frame(..))` must reproduce the
//! normal encoder's packet byte for byte, edited coefficients must
//! reach the decoder, and malformed input must be refused. Pictures
//! are 133x75: not a multiple of 16 (or 8) in either direction, with
//! partial right-edge slices and a partial bottom macroblock row.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame};
use oxideav_prores::decoder::{BitDepth, decode_packet_with_depth};
use oxideav_prores::encoder::quantized::{
    QuantizedFrame, encode_quantized_frame, picture_slice_layout, quantize_frame,
};
use oxideav_prores::encoder::{EncoderConfig, make_encoder_with_config};
use oxideav_prores::frame::{ChromaFormat, Profile};
use oxideav_prores::quant::qscale;

const W: u32 = 133;
const H: u32 = 75;

/// Gradient, hard edges (a bright box and a stripe pattern) and noise,
/// as a little-endian 16-bit plane with samples below `1 << bits`.
fn plane(w: usize, h: usize, bits: u32, seed: u32) -> VideoPlane {
    let max = (1u32 << bits) - 1;
    let mut s = seed;
    let mut data = Vec::with_capacity(w * h * 2);
    for j in 0..h {
        for i in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (s >> 24) % (max / 32 + 1);
            let mut v = (i * max as usize / w + j * max as usize / (2 * h)) as u32 / 2;
            if (w / 3..w / 2).contains(&i) && (h / 4..h / 2).contains(&j) {
                v = max * 9 / 10;
            }
            if j > h * 2 / 3 && (i / 3) % 2 == 0 {
                v = max / 10;
            }
            let v = (v + noise).min(max);
            data.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
    VideoPlane {
        stride: w * 2,
        data,
    }
}

fn frame(chroma: ChromaFormat, bits: u32) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let cw = match chroma {
        ChromaFormat::Y422 => w.div_ceil(2),
        ChromaFormat::Y444 => w,
    };
    VideoFrame {
        pts: Some(0),
        planes: vec![
            plane(w, h, bits, 1),
            plane(cw, h, bits, 2),
            plane(cw, h, bits, 3),
        ],
    }
}

fn params(pix: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("prores"));
    p.media_type = MediaType::Video;
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(pix);
    p
}

fn normal(pix: PixelFormat, config: &EncoderConfig, frame: &VideoFrame) -> Vec<u8> {
    let mut enc = make_encoder_with_config(&params(pix), config.clone()).expect("encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    enc.receive_packet().expect("packet").data
}

struct Case {
    pix: PixelFormat,
    chroma: ChromaFormat,
    depth: BitDepth,
    bits: u32,
    profile: Profile,
}

const HQ_422_10: Case = Case {
    pix: PixelFormat::Yuv422P10Le,
    chroma: ChromaFormat::Y422,
    depth: BitDepth::Ten,
    bits: 10,
    profile: Profile::Hq,
};
const P4444_12: Case = Case {
    pix: PixelFormat::Yuv444P12Le,
    chroma: ChromaFormat::Y444,
    depth: BitDepth::Twelve,
    bits: 12,
    profile: Profile::Prores4444,
};

fn quantize(case: &Case, config: &EncoderConfig, f: &VideoFrame) -> QuantizedFrame {
    quantize_frame(f, W, H, case.depth, case.profile, config).expect("quantize")
}

fn assert_identical(case: &Case, config: EncoderConfig) {
    let f = frame(case.chroma, case.bits);
    let expected = normal(case.pix, &config, &f);
    let q = quantize(case, &config, &f);
    assert_eq!(
        encode_quantized_frame(&config, &q).expect("hook encode"),
        expected,
        "hook packet differs from the normal encoder's"
    );
    // The hook is independent of the thread count too.
    for threads in [2, 5] {
        let multi = config.clone().with_threads(threads);
        assert_eq!(
            encode_quantized_frame(&multi, &q).expect("hook encode"),
            expected
        );
    }
}

#[test]
fn hq_422_10bit_matches_normal_encoder() {
    assert_identical(
        &HQ_422_10,
        EncoderConfig::signature_for_profile(Profile::Hq),
    );
}

#[test]
fn p4444_444_12bit_matches_normal_encoder() {
    assert_identical(
        &P4444_12,
        EncoderConfig::signature_for_profile(Profile::Prores4444),
    );
}

#[test]
fn flat_matrices_and_explicit_carriage_match() {
    assert_identical(&HQ_422_10, EncoderConfig::for_profile(Profile::Hq));
    assert_identical(
        &HQ_422_10,
        EncoderConfig::signature_for_profile(Profile::Hq).with_explicit_qmat_carriage(),
    );
}

#[test]
fn slice_sizes_index_and_padding_match() {
    assert_identical(
        &HQ_422_10,
        EncoderConfig::for_profile(Profile::Hq)
            .with_mbs_per_slice(2)
            .with_quantization_index(9)
            .with_min_frame_size(60_000),
    );
}

#[test]
fn interlaced_matches_normal_encoder() {
    for mode in [1, 2] {
        assert_identical(
            &HQ_422_10,
            EncoderConfig::signature_for_profile(Profile::Hq).with_interlace_mode(mode),
        );
    }
}

#[test]
fn layout_helper_matches_producer() {
    let config = EncoderConfig::for_profile(Profile::Hq).with_interlace_mode(1);
    let q = quantize(&HQ_422_10, &config, &frame(ChromaFormat::Y422, 10));
    let layout = picture_slice_layout(W, H, 1, q.log2_mbs_per_slice);
    assert_eq!(layout.len(), q.pictures.len());
    for (picture, expected) in q.pictures.iter().zip(&layout) {
        let got: Vec<_> = picture
            .slices
            .iter()
            .map(|s| (s.mb_x, s.mb_y, s.mb_count))
            .collect();
        assert_eq!(&got, expected);
    }
}

fn sample(data: &[u8], stride: usize, x: usize, y: usize) -> i32 {
    let o = y * stride + x * 2;
    u16::from_le_bytes([data[o], data[o + 1]]) as i32
}

/// With every AC coefficient zeroed each 8x8 block must decode flat,
/// at about the mean of the original block.
fn assert_dc_only_roundtrip(case: &Case, config: EncoderConfig) {
    let f = frame(case.chroma, case.bits);
    let mut q = quantize(case, &config, &f);
    let full = encode_quantized_frame(&config, &q).unwrap();
    for picture in &mut q.pictures {
        for slice in &mut picture.slices {
            for component in &mut slice.coefficients {
                for block in component.as_chunks_mut::<64>().0.iter_mut() {
                    block[1..].fill(0);
                }
            }
        }
    }
    let dc_only = encode_quantized_frame(&config, &q).unwrap();
    assert!(
        dc_only.len() < full.len() / 2,
        "AC removal must shrink the packet"
    );

    let decoded = decode_packet_with_depth(&dc_only, None, Some((case.depth, case.chroma)))
        .expect("decoder accepts hook output");
    // Reconstruction error of the DC alone: half a step, in samples.
    let w0 = config.quant_matrices.unwrap().luma[0] as f64;
    let step = w0
        * qscale(
            config
                .quantization_index
                .unwrap_or(case.profile.default_quant_index()),
        ) as f64
        / 64.0
        * (1u32 << (case.bits - 9)) as f64;
    let src = &f.planes[0];
    let out = &decoded.planes[0];
    let (w, h) = (W as usize, H as usize);
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let (x1, y1) = ((bx + 8).min(w), (by + 8).min(h));
            let first = sample(&out.data, out.stride, bx, by);
            // Edge blocks replicate the last row/column, so their
            // visible part is flat too, but their mean is of the padded block.
            let mut sum = 0.0;
            for y in by..by + 8 {
                for x in bx..bx + 8 {
                    sum += sample(&src.data, src.stride, x.min(w - 1), y.min(h - 1)) as f64;
                }
            }
            let mean = sum / 64.0;
            for y in by..y1 {
                for x in bx..x1 {
                    assert_eq!(
                        sample(&out.data, out.stride, x, y),
                        first,
                        "block ({bx},{by}) is not flat at ({x},{y})"
                    );
                }
            }
            assert!(
                (first as f64 - mean).abs() <= step / 2.0 + 2.0,
                "block ({bx},{by}): decoded {first}, source mean {mean:.1}, step {step:.1}"
            );
        }
    }
}

#[test]
fn zeroed_ac_decodes_to_flat_blocks_422() {
    assert_dc_only_roundtrip(
        &HQ_422_10,
        EncoderConfig::signature_for_profile(Profile::Hq),
    );
}

#[test]
fn zeroed_ac_decodes_to_flat_blocks_444() {
    assert_dc_only_roundtrip(
        &P4444_12,
        EncoderConfig::signature_for_profile(Profile::Prores4444),
    );
}

#[test]
fn hook_output_decodes_like_normal_output() {
    let config = EncoderConfig::signature_for_profile(Profile::Hq);
    let f = frame(ChromaFormat::Y422, 10);
    let a = decode_packet_with_depth(
        &normal(HQ_422_10.pix, &config, &f),
        None,
        Some((BitDepth::Ten, ChromaFormat::Y422)),
    )
    .unwrap();
    let q = quantize(&HQ_422_10, &config, &f);
    let b = decode_packet_with_depth(
        &encode_quantized_frame(&config, &q).unwrap(),
        None,
        Some((BitDepth::Ten, ChromaFormat::Y422)),
    )
    .unwrap();
    assert_eq!(a.planes.len(), b.planes.len());
    for (pa, pb) in a.planes.iter().zip(&b.planes) {
        assert_eq!(pa.data, pb.data);
    }
}

#[test]
fn malformed_input_is_rejected() {
    let config = EncoderConfig::for_profile(Profile::Hq);
    let q = quantize(&HQ_422_10, &config, &frame(ChromaFormat::Y422, 10));
    let encode = |q: &QuantizedFrame| encode_quantized_frame(&config, q);
    assert!(encode(&q).is_ok());

    let mut bad = q.clone();
    bad.pictures[0].slices[0].coefficients[1].pop();
    assert!(encode(&bad).is_err(), "short chroma vector");

    let mut bad = q.clone();
    bad.pictures[0].slices.pop();
    assert!(encode(&bad).is_err(), "missing slice");

    let mut bad = q.clone();
    bad.pictures[0].slices[1].mb_x += 1;
    assert!(encode(&bad).is_err(), "slice out of place");

    let mut bad = q.clone();
    bad.pictures[0].slices[0].quantization_index = 0;
    assert!(encode(&bad).is_err(), "quantization_index 0");

    let mut bad = q.clone();
    bad.interlace_mode = 1;
    assert!(encode(&bad).is_err(), "picture count vs interlace mode");

    let alpha = config
        .clone()
        .with_alpha_channel_type(oxideav_prores::alpha::AlphaChannelType::Eight);
    assert!(encode_quantized_frame(&alpha, &q).is_err(), "alpha");
}

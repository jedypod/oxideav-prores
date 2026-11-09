//! Coding of caller-supplied, already-quantised DCT coefficients.
//!
//! The normal encoder ([`super::make_encoder`] and the `encode_frame*`
//! functions) turns pixels into a packet in one go: it reads a
//! sample plane, forward-DCTs 8x8 blocks, quantises them, and entropy
//! codes the result. This module splits that pipeline at the
//! quantiser so the first half can run somewhere else (a GPU, another
//! crate, a different machine) while this crate does only the second:
//! entropy coding, slice assembly, and the frame and picture headers.
//!
//! * [`quantize_frame`] is the reference producer: the CPU path from
//!   planes to a [`QuantizedFrame`], making the same choices as the
//!   normal encoder.
//! * [`encode_quantized_frame`] codes a [`QuantizedFrame`] into a
//!   complete RDD 36 `frame()` packet.
//!
//! For equal settings `encode_quantized_frame(quantize_frame(..))` is
//! byte-identical to the normal encoder's packet.
//!
//! # What an external quantiser must produce
//!
//! ## Pictures and slices
//!
//! A frame is one picture (progressive, `interlace_mode == 0`) or two
//! field pictures (`interlace_mode` 1 = top field first, 2 = bottom
//! field first), listed in coding order. The top field holds frame rows
//! 0, 2, 4, ... and has `ceil(height / 2)` rows, the bottom field rows
//! 1, 3, 5, ... with `floor(height / 2)` rows; each field is
//! quantised as if it were its own picture.
//!
//! A picture is cut into 16x16 macroblocks, `ceil(width / 16)` per row
//! and `ceil(picture_height / 16)` rows; the right and bottom edge
//! macroblocks are padded by replicating the last column and row of
//! the picture. Each macroblock row is cut into slices from left to
//! right: as many slices of `1 << log2_mbs_per_slice` macroblocks as
//! fit, then the remainder in decreasing power-of-two sizes (so a row
//! of 25 macroblocks at 8 per slice is 8, 8, 8, 1). Slices are listed
//! row by row. [`picture_slice_layout`] returns exactly the
//! `(mb_x, mb_y, mb_count)` list the coder expects, and
//! [`encode_quantized_frame`] rejects any other list.
//!
//! ## Blocks and coefficient order
//!
//! [`QuantizedSlice::coefficients`] holds one `Vec<i16>` per component
//! (Y, Cb, Cr). Each is a sequence of 8x8 blocks of 64 coefficients,
//! macroblock by macroblock, left to right:
//!
//! * luma: 4 blocks per macroblock, in the order top-left, top-right,
//!   bottom-left, bottom-right (row-major);
//! * chroma 4:2:2 (the chroma plane is half as wide): 2 blocks per
//!   macroblock, top then bottom;
//! * chroma 4:4:4: 4 blocks per macroblock, in the order top-left,
//!   bottom-left, top-right, bottom-right (column-major);
//!
//! Inside a block the 64 coefficients are in **natural (raster)
//! order**: index `v * 8 + u` is the coefficient of horizontal
//! frequency `u` and vertical frequency `v`, so index 0 is DC. They
//! are not zig-zag scanned and DC is not differenced: the crate
//! applies the RDD 36 block scan (progressive or interlaced, chosen by
//! `interlace_mode`), the cross-block slice scan and the DC
//! differencing itself.
//!
//! ## Rounding and scaling
//!
//! Input samples of bit depth `b` are level shifted to
//! `v = s * 2^(9 - b) - 256` (RDD 36 §7.5.1), giving values in
//! `[-256, 256)`. A block is transformed with the orthonormal 8x8
//! DCT-II, `F`, and quantised per RDD 36 §7.3 as
//!
//! ```text
//! QF[k] = round_half_away_from_zero(F[k] * 8 / (W[k] * qScale))
//! ```
//!
//! where `W` is the luma or chroma quantisation matrix in natural
//! order (flat 4s by default; luma matrix for Y, chroma matrix for Cb
//! and Cr) and `qScale = quantization_index` for indices up to 128,
//! `128 + 4 * (quantization_index - 128)` above (Table 15; see
//! [`crate::quant::qscale`]). `round_half_away_from_zero(x)` is
//! `sign(x) * floor(|x| + 0.5)`. The decoder dequantises with
//! `F' = QF * W * qScale / 8`.
//!
//! The CPU reference computes `F` with a single-precision AAN fast DCT
//! and multiplies by a single-precision reciprocal, so a quantiser
//! built differently (another DCT factorisation, different float
//! rounding, a GPU) can differ from the CPU output by one in
//! coefficients that land within float error of a rounding boundary.
//! That is a legitimate encode: any integer coefficient array is a
//! valid stream, and the decoder reconstructs from whatever is coded.
//! Only the byte-identity of the two paths needs the same arithmetic.
//!
//! A block whose 64 samples are all equal has only a DC coefficient
//! (`64 * v` before quantisation); the CPU path takes that shortcut,
//! which agrees with the transform up to float error.
//!
//! ## Choosing `quantization_index`
//!
//! Each slice names its own `quantization_index` (1..=224), which the
//! slice header carries and the decoder uses to dequantise that slice;
//! the coder does not check that the coefficients really were
//! divided by it. The CPU encoder uses one index for every slice of
//! every picture: `EncoderConfig::quantization_index` or else the
//! profile default ([`Profile::default_quant_index`]). With rate
//! control (a bit rate and frame rate in the codec parameters plus
//! `EncoderConfig::rate_control`) the normal encoder instead codes the
//! whole frame repeatedly, binary searching one frame-wide index
//! until the packet is within 5% of the target size; there is no
//! per-slice rate control. [`quantize_frame`] does not do that search,
//! it uses the fixed index. An external quantiser that wants a size
//! target must run its own loop: quantise at an index, call
//! [`encode_quantized_frame`], look at the packet length, repeat.
//! Packet size falls monotonically as the index rises.
//!
//! ## Not supported
//!
//! Alpha (4444 with an alpha channel) is not supported: a config with
//! `alpha_channel_type` set is rejected. Use the normal encoder for
//! alpha.

use oxideav_core::{Error, Result, VideoFrame};

use super::{
    CodedPicture, EncoderConfig, FrameHeaderParams, PictureCoder, assemble_frame, check_dimensions,
    code_slice_blocks, code_slices, mbs_per_slice_to_log2, output_capacity_cap, pad_frame_to_size,
    picture_layout, quant_multipliers, resolve_qmat_carriage, slice_layout,
};
use crate::decoder::BitDepth;
use crate::frame::{ChromaFormat, FrameMeta, Profile};
use crate::quant::QuantMatrices;
use crate::slice::{LUMA_BLOCKS_PER_MB, chroma_blocks_per_mb};

/// Coefficients per 8x8 block.
const BLOCK: usize = 64;

/// One slice of quantised coefficients. See the [module docs](self)
/// for the exact layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedSlice {
    /// Macroblock column of the slice's first macroblock.
    pub mb_x: u32,
    /// Macroblock row of the slice within its picture.
    pub mb_y: u32,
    /// Number of macroblocks in the slice (1, 2, 4 or 8).
    pub mb_count: u8,
    /// `quantization_index` (1..=224) written to the slice header; the
    /// coefficients are expected to be quantised with its `qScale`.
    pub quantization_index: u8,
    /// Y, Cb and Cr coefficients: per component `mb_count * blocks_per_mb`
    /// blocks of 64 natural-order coefficients. `blocks_per_mb` is 4
    /// for luma, 2 (4:2:2) or 4 (4:4:4) for each chroma component.
    pub coefficients: [Vec<i16>; 3],
}

/// The slices of one picture (a whole progressive frame or one field),
/// in coding order: macroblock rows top to bottom, slices left to
/// right.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedPicture {
    pub slices: Vec<QuantizedSlice>,
}

/// A frame of quantised coefficients plus everything the frame and
/// picture headers need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedFrame {
    /// Luma width in samples (1..=65535).
    pub width: u32,
    /// Luma height in samples (1..=65535); for interlaced frames the
    /// height of the whole frame, not of a field.
    pub height: u32,
    pub chroma: ChromaFormat,
    /// 0 progressive, 1 top field first, 2 bottom field first.
    pub interlace_mode: u8,
    /// `log2` of the desired macroblocks per slice (0..=3).
    pub log2_mbs_per_slice: u8,
    /// The matrices the coefficients were quantised with. They are
    /// written to the frame header (in the minimal form, unless
    /// [`EncoderConfig::explicit_qmat_carriage`] asks for both).
    pub quant_matrices: QuantMatrices,
    /// Descriptive header fields (aspect ratio, frame rate code, colour
    /// description).
    pub meta: FrameMeta,
    /// One picture if progressive, else two fields in coding order.
    pub pictures: Vec<QuantizedPicture>,
}

/// `(mb_x, mb_y, mb_count)` of every slice of every picture of a frame,
/// in the order [`encode_quantized_frame`] requires: one list per
/// picture (one for progressive, two for interlaced, in coding order).
pub fn picture_slice_layout(
    width: u32,
    height: u32,
    interlace_mode: u8,
    log2_mbs_per_slice: u8,
) -> Vec<Vec<(u32, u32, u8)>> {
    picture_layout(height as usize, interlace_mode)
        .into_iter()
        .map(|(picture_height, _)| {
            slice_layout(width as usize, picture_height, log2_mbs_per_slice)
                .into_iter()
                .map(|(x, y, n)| (x as u32, y as u32, n as u8))
                .collect()
        })
        .collect()
}

/// Reference producer: forward-DCT and quantise a 3-plane
/// `Yuv4(2|4)4P{,10Le,12Le,16Le}` frame on the CPU, making the choices
/// the normal encoder makes for `config` (matrices, slice size,
/// interlacing, metadata, and a fixed `quantization_index` taken from
/// `config.quantization_index` or `profile.default_quant_index()`).
/// Rate control is not applied. Alpha is not supported.
///
/// `profile` supplies the chroma format and the default index; the
/// profile itself is not stored in the frame (it is signalled by the
/// container fourcc).
pub fn quantize_frame(
    frame: &VideoFrame,
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    profile: Profile,
    config: &EncoderConfig,
) -> Result<QuantizedFrame> {
    reject_alpha(config)?;
    if frame.planes.len() != 3 {
        return Err(Error::invalid(format!(
            "prores quantize_frame: expected 3 planes (got {}); alpha is not supported",
            frame.planes.len()
        )));
    }
    let chroma = profile.chroma_format();
    let quantization_index = config
        .quantization_index
        .unwrap_or_else(|| profile.default_quant_index());
    check_quantization_index(quantization_index)?;
    check_interlace_mode(config.interlace_mode)?;
    check_dimensions(width, height)?;
    let quant_matrices = config.quant_matrices.unwrap_or_default();
    if !quant_matrices.weights_valid() {
        return Err(Error::invalid(
            "prores quantize_frame: quant matrix weight outside RDD 36 range 2..=63",
        ));
    }
    let log2_mbs_per_slice =
        mbs_per_slice_to_log2(config.mbs_per_slice.unwrap_or(super::DEFAULT_MBS_PER_SLICE))?;

    let (w, h) = (width as usize, height as usize);
    let interlaced = config.interlace_mode != 0;
    let mut pictures = Vec::new();
    for (picture_height, field) in picture_layout(h, config.interlace_mode) {
        let coder = PictureCoder {
            frame,
            frame_w: w,
            frame_h: h,
            c_w: match chroma {
                ChromaFormat::Y422 => w.div_ceil(2),
                ChromaFormat::Y444 => w,
            },
            chroma,
            bit_depth,
            quantization_index,
            luma_mul: quant_multipliers(&quant_matrices.luma, quantization_index),
            chroma_mul: quant_multipliers(&quant_matrices.chroma, quantization_index),
            alpha: None,
            interlaced,
            field,
        };
        let mut blocks = Vec::new();
        let slices = slice_layout(w, picture_height, log2_mbs_per_slice)
            .into_iter()
            .map(|(mx, my, mbs)| {
                coder.quantize_slice(mx, my, mbs, &mut blocks);
                QuantizedSlice {
                    mb_x: mx as u32,
                    mb_y: my as u32,
                    mb_count: mbs as u8,
                    quantization_index,
                    coefficients: split_components(&blocks, mbs, chroma),
                }
            })
            .collect();
        pictures.push(QuantizedPicture { slices });
    }
    Ok(QuantizedFrame {
        width,
        height,
        chroma,
        interlace_mode: config.interlace_mode,
        log2_mbs_per_slice,
        quant_matrices,
        meta: config.meta.unwrap_or_default(),
        pictures,
    })
}

/// Code a [`QuantizedFrame`] into a complete RDD 36 `frame()` packet:
/// frame header, picture headers, slice tables, and entropy-coded
/// slices.
///
/// From `config` only `threads` (slices of a picture are coded in
/// parallel; the output does not depend on the thread count),
/// `min_frame_size` (stuffing) and `explicit_qmat_carriage` are used;
/// everything else comes from the frame. An `alpha_channel_type` in
/// `config` is an error. The slice list of each picture must match
/// [`picture_slice_layout`] and every coefficient vector must have
/// the length its slice implies, otherwise `Error::invalid` is
/// returned. Coefficient magnitudes are not range checked beyond what
/// `i16` holds; a conforming quantiser stays within about +-8192.
pub fn encode_quantized_frame(config: &EncoderConfig, frame: &QuantizedFrame) -> Result<Vec<u8>> {
    reject_alpha(config)?;
    check_dimensions(frame.width, frame.height)?;
    check_interlace_mode(frame.interlace_mode)?;
    if frame.log2_mbs_per_slice > 3 {
        return Err(Error::invalid(
            "prores encode_quantized_frame: log2_mbs_per_slice must be 0..=3",
        ));
    }
    if !frame.quant_matrices.weights_valid() {
        return Err(Error::invalid(
            "prores encode_quantized_frame: quant matrix weight outside RDD 36 range 2..=63",
        ));
    }
    let layout = picture_layout(frame.height as usize, frame.interlace_mode);
    if frame.pictures.len() != layout.len() {
        return Err(Error::invalid(format!(
            "prores encode_quantized_frame: interlace_mode {} needs {} picture(s), got {}",
            frame.interlace_mode,
            layout.len(),
            frame.pictures.len()
        )));
    }

    let threads = config.threads.max(1);
    let interlaced = frame.interlace_mode != 0;
    let mut coded = Vec::with_capacity(layout.len());
    for (picture, &(picture_height, _)) in frame.pictures.iter().zip(&layout) {
        let expected = slice_layout(
            frame.width as usize,
            picture_height,
            frame.log2_mbs_per_slice,
        );
        if picture.slices.len() != expected.len() {
            return Err(Error::invalid(format!(
                "prores encode_quantized_frame: picture has {} slices, layout needs {}",
                picture.slices.len(),
                expected.len()
            )));
        }
        for (slice, &(mx, my, mbs)) in picture.slices.iter().zip(&expected) {
            check_slice(slice, (mx, my, mbs), frame.chroma)?;
        }
        let runs = code_slices(&picture.slices, threads, |slice, scratch, out| {
            join_components(
                &slice.coefficients,
                slice.mb_count as usize,
                frame.chroma,
                &mut scratch.blocks,
            );
            code_slice_blocks(
                frame.chroma,
                interlaced,
                slice.quantization_index,
                slice.mb_count as usize,
                None::<fn(&mut Vec<u16>, &mut Vec<u8>) -> Result<()>>,
                scratch,
                out,
            )
        })?;
        coded.push(CodedPicture {
            runs,
            slice_count: picture.slices.len(),
            log2_slice_mb_width: frame.log2_mbs_per_slice,
        });
    }

    let (_, load_luma, load_chroma) =
        resolve_qmat_carriage(Some(frame.quant_matrices), config.explicit_qmat_carriage);
    let data = assemble_frame(
        &coded,
        FrameHeaderParams {
            width: frame.width as u16,
            height: frame.height as u16,
            chroma: frame.chroma,
            interlace_mode: frame.interlace_mode,
            qmats: frame.quant_matrices,
            load_luma,
            load_chroma,
            alpha_channel_type: 0,
            meta: frame.meta,
        },
        output_capacity_cap(frame.width as u16, frame.height as u16, frame.chroma),
    )?;
    match config.min_frame_size {
        Some(min) => pad_frame_to_size(&data, min),
        None => Ok(data),
    }
}

fn reject_alpha(config: &EncoderConfig) -> Result<()> {
    if config.alpha_channel_type.is_some() {
        return Err(Error::unsupported(
            "prores quantized hook: alpha is not supported",
        ));
    }
    Ok(())
}

fn check_quantization_index(qi: u8) -> Result<()> {
    if !(1..=224).contains(&qi) {
        return Err(Error::invalid(
            "prores quantized hook: quantization_index must be 1..=224 (RDD 36 Table 15)",
        ));
    }
    Ok(())
}

fn check_interlace_mode(mode: u8) -> Result<()> {
    if mode > 2 {
        return Err(Error::invalid(
            "prores quantized hook: interlace_mode must be 0, 1 or 2",
        ));
    }
    Ok(())
}

fn check_slice(
    slice: &QuantizedSlice,
    (mx, my, mbs): (usize, usize, usize),
    chroma: ChromaFormat,
) -> Result<()> {
    if slice.mb_x as usize != mx || slice.mb_y as usize != my || slice.mb_count as usize != mbs {
        return Err(Error::invalid(format!(
            "prores encode_quantized_frame: slice at ({}, {}) x{} where the layout \
             has ({mx}, {my}) x{mbs}",
            slice.mb_x, slice.mb_y, slice.mb_count
        )));
    }
    check_quantization_index(slice.quantization_index)?;
    let chroma_blocks = chroma_blocks_per_mb(chroma);
    let want = [
        mbs * LUMA_BLOCKS_PER_MB * BLOCK,
        mbs * chroma_blocks * BLOCK,
        mbs * chroma_blocks * BLOCK,
    ];
    for (c, (have, want)) in slice.coefficients.iter().zip(want).enumerate() {
        if have.len() != want {
            return Err(Error::invalid(format!(
                "prores encode_quantized_frame: slice ({mx}, {my}) component {c} has {} \
                 coefficients, expected {want}",
                have.len()
            )));
        }
    }
    Ok(())
}

/// Slice-order blocks (per macroblock: Y blocks, Cb blocks, Cr blocks)
/// to one coefficient vector per component.
fn split_components(blocks: &[[i32; 64]], mbs: usize, chroma: ChromaFormat) -> [Vec<i16>; 3] {
    let cb = chroma_blocks_per_mb(chroma);
    let per_mb = LUMA_BLOCKS_PER_MB + 2 * cb;
    let component = |offset: usize, count: usize| {
        let mut v = Vec::with_capacity(mbs * count * BLOCK);
        for mb in 0..mbs {
            for block in &blocks[mb * per_mb + offset..mb * per_mb + offset + count] {
                v.extend(block.iter().map(|&c| c as i16));
            }
        }
        v
    };
    [
        component(0, LUMA_BLOCKS_PER_MB),
        component(LUMA_BLOCKS_PER_MB, cb),
        component(LUMA_BLOCKS_PER_MB + cb, cb),
    ]
}

/// Inverse of [`split_components`], into the scratch block list.
fn join_components(
    coefficients: &[Vec<i16>; 3],
    mbs: usize,
    chroma: ChromaFormat,
    blocks: &mut Vec<[i32; 64]>,
) {
    let cb = chroma_blocks_per_mb(chroma);
    blocks.clear();
    for mb in 0..mbs {
        for (component, count) in [(0, LUMA_BLOCKS_PER_MB), (1, cb), (2, cb)] {
            let src = &coefficients[component][mb * count * BLOCK..(mb + 1) * count * BLOCK];
            for chunk in src.as_chunks::<BLOCK>().0 {
                blocks.push(std::array::from_fn(|k| chunk[k] as i32));
            }
        }
    }
}

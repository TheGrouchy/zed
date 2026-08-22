// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::{self, Debug},
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A cardinal direction for a CSS-compatible linear alpha mask.
///
/// The locked Waypath masks use only `to top`, `to right`, and `to bottom`.
/// Keeping the direction finite prevents an arbitrary-angle implementation
/// from being presented as CSS-exact before it has renderer conformance.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum LinearGradientMaskDirection {
    /// The gradient progresses from the bottom edge to the top edge.
    ToTop = 0,
    /// The gradient progresses from the left edge to the right edge.
    ToRight = 1,
    /// The gradient progresses from the top edge to the bottom edge.
    #[default]
    ToBottom = 2,
    /// The gradient progresses from the right edge to the left edge.
    ToLeft = 3,
}

/// A stop in a linear alpha mask.
///
/// `percentage * mask_axis_length + offset` represents both percentages and
/// source values such as `calc(100% - 30px)` without resolving them early.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct LinearGradientMaskStop {
    /// The mask alpha at this stop, in the inclusive range 0..=1.
    pub alpha: f32,
    /// Percentage of the mask axis, in the inclusive range 0..=1.
    pub percentage: f32,
    /// Pixel offset added after resolving `percentage`.
    pub offset: Pixels,
}

/// The largest stop list represented by the native mask shader ABI.
pub const MAX_LINEAR_GRADIENT_MASK_STOPS: usize = 5;

/// Maximum one-sided text-shadow convolution radius accepted by the native
/// renderer. The fixed ABI keeps shader loops and structured-buffer layout
/// identical across DirectX, WGPU, and Metal.
pub const MAX_TEXT_SHADOW_KERNEL_RADIUS: usize = 63;

/// One CSS-compatible text shadow in logical pixels.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct TextShadow {
    /// Translation applied after blurring the glyph alpha mask.
    pub offset: Point<Pixels>,
    /// CSS blur radius. Blink resolves this to `sigma = radius / 2`.
    pub blur_radius: Pixels,
    /// Shadow color, including its independent alpha.
    pub color: Hsla,
}

/// GPU-ready text shadow with the exact discrete one-dimensional kernel.
/// The renderer applies it horizontally and vertically with 8-bit rounding
/// between passes, matching Chromium's A8 mask contract.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C, align(8))]
#[expect(missing_docs)]
pub struct ScaledTextShadow {
    pub source_bounds: Bounds<ScaledPixels>,
    pub first_pass_bounds: Bounds<ScaledPixels>,
    pub final_bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub offset: Point<ScaledPixels>,
    pub blur_radius: f32,
    pub kernel_radius: u32,
    pub normalization_weight: u32,
    pub normalization_pad: u32,
    pub color: Hsla,
    pub kernel: [f32; MAX_TEXT_SHADOW_KERNEL_RADIUS + 1],
}

/// A failure that prevents text-shadow source alpha from being represented
/// without silently changing CSS semantics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[expect(missing_docs)]
pub enum TextShadowGroupError {
    NoActiveGroup,
    NestedGroup,
    LinearMaskConflict,
    EmptyShadowList,
    NonFiniteValue,
    NegativeBlurRadius,
    FractionalDeviceOffset,
    KernelTooWide,
    EmptyGlyphSource,
    ColorGlyph,
    TransformedGlyph,
    UnbalancedLayer,
    UnsupportedForegroundPrimitive,
}

impl fmt::Display for TextShadowGroupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NoActiveGroup => "no text-shadow group is active",
            Self::NestedGroup => "nested text-shadow groups are unsupported",
            Self::LinearMaskConflict => {
                "text-shadow and linear-mask offscreen groups cannot be nested"
            }
            Self::EmptyShadowList => "text-shadow group has no shadows",
            Self::NonFiniteValue => "text-shadow contains a non-finite value",
            Self::NegativeBlurRadius => "text-shadow blur radius is negative",
            Self::FractionalDeviceOffset => {
                "text-shadow offset is fractional in device pixels and is not certified"
            }
            Self::KernelTooWide => "text-shadow blur kernel exceeds the native ABI",
            Self::EmptyGlyphSource => "text-shadow group contains no monochrome glyphs",
            Self::ColorGlyph => "text-shadow contains an unsupported color-font glyph",
            Self::TransformedGlyph => {
                "text-shadow contains a glyph transform not certified by this path"
            }
            Self::UnbalancedLayer => "text-shadow group has an unbalanced paint layer",
            Self::UnsupportedForegroundPrimitive => {
                "text-shadow group contains a non-text foreground primitive"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for TextShadowGroupError {}

/// A single CSS-default linear mask layer over an explicit mask border box.
///
/// This type intentionally represents only the locked longhand defaults:
/// `mask-size:auto`, `mask-position:0% 0%`, `mask-repeat:repeat`,
/// `mask-origin:border-box`, `mask-clip:border-box`, `mask-composite:add`, and
/// alpha mode for a generated gradient image. Other mask layers cannot be
/// constructed through this API and therefore fail closed at the adapter.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct LinearGradientMask {
    bounds: Bounds<Pixels>,
    direction: LinearGradientMaskDirection,
    stop_count: u32,
    stops: [LinearGradientMaskStop; MAX_LINEAR_GRADIENT_MASK_STOPS],
    pad: u32,
}

impl LinearGradientMask {
    /// Creates a validated cardinal linear alpha mask.
    pub fn try_new(
        bounds: Bounds<Pixels>,
        direction: LinearGradientMaskDirection,
        stops: &[LinearGradientMaskStop],
    ) -> Result<Self, LinearGradientMaskError> {
        if !bounds.origin.x.0.is_finite()
            || !bounds.origin.y.0.is_finite()
            || !bounds.size.width.0.is_finite()
            || !bounds.size.height.0.is_finite()
            || bounds.size.width.0 <= 0.0
            || bounds.size.height.0 <= 0.0
        {
            return Err(LinearGradientMaskError::InvalidBounds);
        }
        if !(2..=MAX_LINEAR_GRADIENT_MASK_STOPS).contains(&stops.len()) {
            return Err(LinearGradientMaskError::InvalidStopCount(stops.len()));
        }

        let axis_length = match direction {
            LinearGradientMaskDirection::ToTop | LinearGradientMaskDirection::ToBottom => {
                bounds.size.height.0
            }
            LinearGradientMaskDirection::ToRight | LinearGradientMaskDirection::ToLeft => {
                bounds.size.width.0
            }
        };
        let mut previous_position = f32::NEG_INFINITY;
        for (index, stop) in stops.iter().enumerate() {
            if !stop.alpha.is_finite()
                || !(0.0..=1.0).contains(&stop.alpha)
                || !stop.percentage.is_finite()
                || !(0.0..=1.0).contains(&stop.percentage)
                || !stop.offset.0.is_finite()
            {
                return Err(LinearGradientMaskError::InvalidStop(index));
            }
            let position = stop.percentage * axis_length + stop.offset.0;
            if position < previous_position {
                return Err(LinearGradientMaskError::StopsOutOfOrder(index));
            }
            previous_position = position;
        }

        let mut stored_stops = [LinearGradientMaskStop::default(); MAX_LINEAR_GRADIENT_MASK_STOPS];
        stored_stops[..stops.len()].copy_from_slice(stops);
        Ok(Self {
            bounds,
            direction,
            stop_count: stops.len() as u32,
            stops: stored_stops,
            pad: 0,
        })
    }

    /// Evaluates the exact scalar alpha ramp used by each renderer shader.
    pub fn alpha_at(&self, position: Point<Pixels>) -> f32 {
        if position.x < self.bounds.left()
            || position.x > self.bounds.right()
            || position.y < self.bounds.top()
            || position.y > self.bounds.bottom()
        {
            return 0.0;
        }

        let (axis_position, axis_length) = match self.direction {
            LinearGradientMaskDirection::ToTop => (
                self.bounds.bottom().0 - position.y.0,
                self.bounds.size.height.0,
            ),
            LinearGradientMaskDirection::ToRight => (
                position.x.0 - self.bounds.left().0,
                self.bounds.size.width.0,
            ),
            LinearGradientMaskDirection::ToBottom => (
                position.y.0 - self.bounds.top().0,
                self.bounds.size.height.0,
            ),
            LinearGradientMaskDirection::ToLeft => (
                self.bounds.right().0 - position.x.0,
                self.bounds.size.width.0,
            ),
        };
        let stops = &self.stops[..self.stop_count as usize];
        let resolved =
            |stop: &LinearGradientMaskStop| stop.percentage * axis_length + stop.offset.0;
        if axis_position < resolved(&stops[0]) {
            return stops[0].alpha;
        }
        let mut index = 0;
        while index + 1 < stops.len() && axis_position >= resolved(&stops[index + 1]) {
            index += 1;
        }
        if index + 1 == stops.len() {
            return stops[index].alpha;
        }
        let start = resolved(&stops[index]);
        let end = resolved(&stops[index + 1]);
        if end == start {
            return stops[index + 1].alpha;
        }
        let factor = ((axis_position - start) / (end - start)).clamp(0.0, 1.0);
        stops[index].alpha + (stops[index + 1].alpha - stops[index].alpha) * factor
    }

    pub(crate) fn scale(self, factor: f32) -> LinearGradientMaskParams {
        let mut stops = [ScaledLinearGradientMaskStop::default(); MAX_LINEAR_GRADIENT_MASK_STOPS];
        for (target, source) in stops.iter_mut().zip(self.stops) {
            *target = ScaledLinearGradientMaskStop {
                alpha: source.alpha,
                percentage: source.percentage,
                offset: ScaledPixels(source.offset.0 * factor),
            };
        }
        LinearGradientMaskParams {
            bounds: self.bounds.scale(factor),
            direction: self.direction,
            stop_count: self.stop_count,
            stops,
            pad: 0,
        }
    }
}

/// Validation failure for a native linear gradient mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LinearGradientMaskError {
    /// The mask bounds are empty or non-finite.
    InvalidBounds,
    /// The native ABI requires two through five stops.
    InvalidStopCount(usize),
    /// A stop contains a non-finite or out-of-range value.
    InvalidStop(usize),
    /// Resolved stop positions are not monotonically nondecreasing.
    StopsOutOfOrder(usize),
}

impl fmt::Display for LinearGradientMaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds => {
                write!(formatter, "linear mask bounds must be finite and positive")
            }
            Self::InvalidStopCount(count) => {
                write!(
                    formatter,
                    "linear mask requires two through five stops, got {count}"
                )
            }
            Self::InvalidStop(index) => write!(formatter, "linear mask stop {index} is invalid"),
            Self::StopsOutOfOrder(index) => {
                write!(
                    formatter,
                    "linear mask stop {index} resolves before its predecessor"
                )
            }
        }
    }
}

impl std::error::Error for LinearGradientMaskError {}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct ScaledLinearGradientMaskStop {
    pub alpha: f32,
    pub percentage: f32,
    pub offset: ScaledPixels,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C, align(8))]
#[expect(missing_docs)]
pub struct LinearGradientMaskParams {
    pub bounds: Bounds<ScaledPixels>,
    pub direction: LinearGradientMaskDirection,
    pub stop_count: u32,
    pub stops: [ScaledLinearGradientMaskStop; MAX_LINEAR_GRADIENT_MASK_STOPS],
    pub pad: u32,
}

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

fn normalized_discrete_gaussian(sigma: f64) -> Vec<f32> {
    const GOOD_ENOUGH: f64 = 0.01;

    fn bessel_i0(value: f64) -> f64 {
        let squared_over_four = value * value / 4.0;
        let mut sum = 1.0;
        let mut factor = 1.0;
        let mut k = 1.0;
        while factor > 1.0 / 1_000_000.0 {
            factor *= squared_over_four / (k * k);
            sum += factor;
            k += 1.0;
        }
        sum
    }

    fn bessel_i1(value: f64) -> f64 {
        let squared_over_four = value * value / 4.0;
        let mut sum = value / 2.0;
        let mut factor = sum;
        let mut k = 1.0;
        while factor > 1.0 / 1_000_000.0 {
            factor *= squared_over_four / (k * (k + 1.0));
            sum += factor;
            k += 1.0;
        }
        sum
    }

    let variance = sigma * sigma;
    let denominator = variance.exp();
    let mut bessel = [0.0_f64; 6];
    let mut gaussian = [0.0_f64; 6];
    bessel[0] = bessel_i0(variance);
    bessel[1] = bessel_i1(variance);
    gaussian[0] = bessel[0] / denominator;
    gaussian[1] = bessel[1] / denominator;
    let mut count = 1;
    while gaussian[count] > GOOD_ENOUGH {
        bessel[count + 1] = -(2.0 * count as f64 / variance) * bessel[count] + bessel[count - 1];
        gaussian[count + 1] = bessel[count + 1] / denominator;
        count += 1;
    }

    let mut sum = gaussian[0];
    for value in gaussian[1..count].iter().rev() {
        sum += 2.0 * value;
    }
    for value in &mut gaussian[..count] {
        *value /= sum;
    }
    let side_sum = gaussian[1..count].iter().rev().sum::<f64>() * 2.0;
    gaussian[0] = 1.0 - side_sum;

    gaussian[..count]
        .iter()
        // Skia converts each coefficient to unsigned 0.16 fixed point.
        .map(|value| ((value * 65_536.0).round() / 65_536.0) as f32)
        .collect()
}

struct TextShadowKernel {
    values: Vec<f32>,
    normalization_weight: u32,
}

fn convolved_box_kernel(sigma: f64) -> TextShadowKernel {
    let window = ((sigma * 3.0 * (2.0 * std::f64::consts::PI).sqrt() / 4.0) + 0.5)
        .floor()
        .max(1.0) as usize;
    let widths = [
        window,
        window,
        if window % 2 == 1 { window } else { window + 1 },
    ];
    let mut coefficients = vec![1_u64];
    for width in widths {
        let mut next = vec![0_u64; coefficients.len() + width - 1];
        for (index, coefficient) in coefficients.iter().enumerate() {
            for target in &mut next[index..index + width] {
                *target += coefficient;
            }
        }
        coefficients = next;
    }
    debug_assert_eq!(coefficients.len() % 2, 1);
    let divisor = widths.iter().product::<usize>() as f64;
    let normalization_weight = ((u32::MAX as f64 + 1.0) / divisor).round() as u32;
    let center = coefficients.len() / 2;
    TextShadowKernel {
        // These integer convolution coefficients are exactly representable in
        // f32 for the bounded ABI. Shaders apply Skia's 0.32 normalization
        // with an emulated rounded multiply-high instead of losing one A8 bit.
        values: coefficients[center..]
            .iter()
            .map(|coefficient| *coefficient as f32)
            .collect(),
        normalization_weight,
    }
}

fn text_shadow_kernel(blur_radius: f32) -> Result<TextShadowKernel, TextShadowGroupError> {
    if !blur_radius.is_finite() {
        return Err(TextShadowGroupError::NonFiniteValue);
    }
    if blur_radius < 0.0 {
        return Err(TextShadowGroupError::NegativeBlurRadius);
    }
    if blur_radius == 0.0 {
        return Ok(TextShadowKernel {
            values: vec![1.0],
            normalization_weight: 0,
        });
    }
    let sigma = f64::from(blur_radius) / 2.0;
    if sigma <= 1.0 / 3.0 {
        return Ok(TextShadowKernel {
            values: vec![1.0],
            normalization_weight: 0,
        });
    }
    let kernel = if sigma < 2.0 {
        TextShadowKernel {
            values: normalized_discrete_gaussian(sigma),
            normalization_weight: 0,
        }
    } else {
        convolved_box_kernel(sigma)
    };
    if kernel.values.len() > MAX_TEXT_SHADOW_KERNEL_RADIUS + 1 {
        return Err(TextShadowGroupError::KernelTooWide);
    }
    Ok(kernel)
}

impl TextShadow {
    fn scale(
        self,
        factor: f32,
        source_bounds: Bounds<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
    ) -> Result<ScaledTextShadow, TextShadowGroupError> {
        if !factor.is_finite()
            || !self.offset.x.as_f32().is_finite()
            || !self.offset.y.as_f32().is_finite()
            || ![self.color.h, self.color.s, self.color.l, self.color.a]
                .into_iter()
                .all(f32::is_finite)
        {
            return Err(TextShadowGroupError::NonFiniteValue);
        }
        let blur_radius = self.blur_radius.as_f32() * factor;
        let weights = text_shadow_kernel(blur_radius)?;
        let kernel_radius = weights.values.len() - 1;
        let radius = ScaledPixels(kernel_radius as f32);
        // Skia's SIMD small-blur path runs vertical then horizontal. Its
        // three-box path runs horizontal then vertical. A8 rounding occurs
        // between passes, so the order is observable and must be preserved.
        let small_blur = blur_radius < 4.0;
        let first_pass_bounds = if small_blur {
            Bounds {
                origin: point(source_bounds.origin.x, source_bounds.origin.y - radius),
                size: Size {
                    width: source_bounds.size.width,
                    height: source_bounds.size.height + radius + radius,
                },
            }
        } else {
            Bounds {
                origin: point(source_bounds.origin.x - radius, source_bounds.origin.y),
                size: Size {
                    width: source_bounds.size.width + radius + radius,
                    height: source_bounds.size.height,
                },
            }
        };
        let offset = self.offset.scale(factor);
        if offset.x.0.fract() != 0.0 || offset.y.0.fract() != 0.0 {
            return Err(TextShadowGroupError::FractionalDeviceOffset);
        }
        let final_bounds = if small_blur {
            Bounds {
                origin: point(
                    first_pass_bounds.origin.x - radius + offset.x,
                    first_pass_bounds.origin.y + offset.y,
                ),
                size: Size {
                    width: first_pass_bounds.size.width + radius + radius,
                    height: first_pass_bounds.size.height,
                },
            }
        } else {
            Bounds {
                origin: point(
                    first_pass_bounds.origin.x + offset.x,
                    first_pass_bounds.origin.y - radius + offset.y,
                ),
                size: Size {
                    width: first_pass_bounds.size.width,
                    height: first_pass_bounds.size.height + radius + radius,
                },
            }
        };
        let mut kernel = [0.0; MAX_TEXT_SHADOW_KERNEL_RADIUS + 1];
        kernel[..weights.values.len()].copy_from_slice(&weights.values);
        Ok(ScaledTextShadow {
            source_bounds,
            first_pass_bounds,
            final_bounds,
            content_mask,
            offset,
            blur_radius,
            kernel_radius: kernel_radius as u32,
            normalization_weight: weights.normalization_weight,
            normalization_pad: 0,
            color: self.color,
            kernel,
        })
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    layer_underflowed: bool,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    /// Backdrop-blur regions — deliberately OUTSIDE the primitive batch
    /// stream: the renderer breaks its render pass at each blur's order to
    /// snapshot the framebuffer (macOS Metal; other renderers ignore them).
    pub backdrop_blurs: Vec<BackdropBlur>,
    /// Atomic subtree groups whose children must be flattened before the
    /// linear alpha mask is applied once to the group result.
    pub linear_gradient_mask_groups: Vec<LinearGradientMaskGroup>,
    active_linear_gradient_mask_group: Option<ActiveLinearGradientMaskGroup>,
    /// Glyph-alpha groups rendered behind their corresponding source text.
    pub text_shadow_groups: Vec<TextShadowGroup>,
    active_text_shadow_group: Option<ActiveTextShadowGroup>,
}

#[expect(missing_docs)]
pub struct LinearGradientMaskGroup {
    pub order: DrawOrder,
    pub mask: LinearGradientMaskParams,
    pub scene: Box<Scene>,
}

struct ActiveLinearGradientMaskGroup {
    order: DrawOrder,
    mask: LinearGradientMaskParams,
    scene: Box<Scene>,
    paint_operation_start: usize,
    color_glyph_attempted: bool,
}

#[expect(missing_docs)]
pub struct TextShadowGroup {
    pub order: DrawOrder,
    /// Normally rendered source text, including its platform LCD mode.
    pub scene: Box<Scene>,
    /// Independent grayscale coverage used as the shadow mask.
    pub shadow_scene: Box<Scene>,
    pub shadows: Vec<ScaledTextShadow>,
}

struct ActiveTextShadowGroup {
    order: DrawOrder,
    shadows: Vec<TextShadow>,
    scale_factor: f32,
    content_mask: ContentMask<ScaledPixels>,
    scene: Box<Scene>,
    shadow_scene: Box<Scene>,
    paint_operation_start: usize,
    color_glyph_attempted: bool,
    transformed_glyph_attempted: bool,
}

/// A failure that prevents a subtree mask from being represented exactly.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LinearGradientMaskGroupError {
    /// No group is active at the matching pop boundary.
    NoActiveGroup,
    /// CSS mask groups cannot be flattened recursively by this bounded path.
    NestedGroup,
    /// Text shadows require their own offscreen pass and cannot share this
    /// bounded single-level group path.
    TextShadowConflict,
    /// A layer opened inside the group was not closed inside the group.
    UnbalancedLayer,
    /// Backdrop sampling needs a separate, explicitly defined group contract.
    BackdropBlur,
    /// Platform video surfaces cannot be sampled into the mask texture.
    Surface,
    /// A cached or manually inserted LCD sprite bypassed grayscale glyph rasterization.
    UnconvertedSubpixelText,
    /// Color-font glyphs need a separately certified alpha/color contract.
    ColorGlyph,
}

impl fmt::Display for LinearGradientMaskGroupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActiveGroup => write!(formatter, "no linear mask group is active"),
            Self::NestedGroup => write!(formatter, "nested linear mask groups are unsupported"),
            Self::TextShadowConflict => write!(
                formatter,
                "linear mask and text-shadow offscreen groups cannot be nested"
            ),
            Self::UnbalancedLayer => write!(formatter, "linear mask group has an unbalanced layer"),
            Self::BackdropBlur => write!(formatter, "linear mask group contains a backdrop blur"),
            Self::Surface => write!(formatter, "linear mask group contains a platform surface"),
            Self::UnconvertedSubpixelText => {
                write!(
                    formatter,
                    "linear mask group contains an unconverted LCD subpixel sprite"
                )
            }
            Self::ColorGlyph => {
                write!(
                    formatter,
                    "linear mask group contains an unsupported color-font glyph"
                )
            }
        }
    }
}

impl std::error::Error for LinearGradientMaskGroupError {}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.layer_underflowed = false;
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.backdrop_blurs.clear();
        self.linear_gradient_mask_groups.clear();
        self.active_linear_gradient_mask_group = None;
        self.text_shadow_groups.clear();
        self.active_text_shadow_group = None;
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        if let Some(group) = self.active_linear_gradient_mask_group.as_mut() {
            group.scene.push_layer(bounds);
            self.paint_operations
                .push(PaintOperation::StartLayer(bounds));
            return;
        }
        if let Some(group) = self.active_text_shadow_group.as_mut() {
            group.scene.push_layer(bounds);
            self.paint_operations
                .push(PaintOperation::StartLayer(bounds));
            return;
        }
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        if let Some(group) = self.active_linear_gradient_mask_group.as_mut() {
            group.scene.pop_layer();
            self.paint_operations.push(PaintOperation::EndLayer);
            return;
        }
        if let Some(group) = self.active_text_shadow_group.as_mut() {
            group.scene.pop_layer();
            self.paint_operations.push(PaintOperation::EndLayer);
            return;
        }
        if self.layer_stack.pop().is_none() {
            self.layer_underflowed = true;
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_backdrop_blur(&mut self, mut blur: BackdropBlur) {
        if let Some(group) = self.active_linear_gradient_mask_group.as_mut() {
            let previous_len = group.scene.len();
            group.scene.insert_backdrop_blur(blur);
            if group.scene.len() != previous_len {
                self.paint_operations
                    .push(PaintOperation::BackdropBlur(blur));
            }
            return;
        }
        if let Some(group) = self.active_text_shadow_group.as_mut() {
            let previous_len = group.scene.len();
            group.scene.insert_backdrop_blur(blur);
            if group.scene.len() != previous_len {
                self.paint_operations
                    .push(PaintOperation::BackdropBlur(blur));
            }
            return;
        }
        let clipped_bounds = blur.bounds.intersect(&blur.content_mask.bounds);
        if clipped_bounds.is_empty() {
            return;
        }
        blur.order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        self.backdrop_blurs.push(blur);
        self.paint_operations
            .push(PaintOperation::BackdropBlur(blur));
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        if let Some(group) = self.active_linear_gradient_mask_group.as_mut() {
            let previous_len = group.scene.len();
            group.scene.insert_primitive(primitive.clone());
            if group.scene.len() != previous_len {
                self.paint_operations
                    .push(PaintOperation::Primitive(primitive));
            }
            return;
        }
        if let Some(group) = self.active_text_shadow_group.as_mut() {
            let previous_len = group.scene.len();
            group.scene.insert_primitive(primitive.clone());
            if group.scene.len() != previous_len {
                self.paint_operations
                    .push(PaintOperation::Primitive(primitive));
            }
            return;
        }
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    /// Begins an atomic subtree whose fully composited pixels will receive one
    /// linear alpha mask. Nested groups fail closed.
    pub fn push_linear_gradient_mask_group(
        &mut self,
        mask: LinearGradientMaskParams,
    ) -> Result<(), LinearGradientMaskGroupError> {
        if self.active_linear_gradient_mask_group.is_some() {
            return Err(LinearGradientMaskGroupError::NestedGroup);
        }
        if self.active_text_shadow_group.is_some() {
            return Err(LinearGradientMaskGroupError::TextShadowConflict);
        }
        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(mask.bounds));
        let paint_operation_start = self.paint_operations.len();
        self.paint_operations
            .push(PaintOperation::StartLinearGradientMaskGroup(mask));
        self.active_linear_gradient_mask_group = Some(ActiveLinearGradientMaskGroup {
            order,
            mask,
            scene: Box::default(),
            paint_operation_start,
            color_glyph_attempted: false,
        });
        Ok(())
    }

    /// Whether new monochrome glyphs must use grayscale alpha rasterization.
    ///
    /// Paint-cache replay can still introduce a previously rasterized LCD
    /// sprite. The matching pop validation rejects that stale representation.
    pub(crate) fn has_active_linear_gradient_mask_group(&self) -> bool {
        self.active_linear_gradient_mask_group.is_some()
    }

    /// Records an attempted color-font glyph so group completion can discard
    /// the entire subtree instead of silently treating it as a normal image.
    pub(crate) fn mark_color_glyph_in_linear_gradient_mask_group(&mut self) {
        if let Some(group) = self.active_linear_gradient_mask_group.as_mut() {
            group.color_glyph_attempted = true;
        }
    }

    /// Begins collecting an independent grayscale glyph mask. Normally
    /// rendered glyphs are captured in an atomic foreground scene so their
    /// platform-selected LCD mode is preserved and they cannot escape a
    /// subsequently rejected shadow group.
    pub fn push_text_shadow_group(
        &mut self,
        bounds: Bounds<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
        scale_factor: f32,
        shadows: Vec<TextShadow>,
    ) -> Result<(), TextShadowGroupError> {
        if self.active_text_shadow_group.is_some() {
            return Err(TextShadowGroupError::NestedGroup);
        }
        if self.active_linear_gradient_mask_group.is_some() {
            return Err(TextShadowGroupError::LinearMaskConflict);
        }
        if shadows.is_empty() {
            return Err(TextShadowGroupError::EmptyShadowList);
        }
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return Err(TextShadowGroupError::NonFiniteValue);
        }
        for shadow in &shadows {
            shadow.scale(scale_factor, bounds, content_mask)?;
        }
        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(bounds));
        let paint_operation_start = self.paint_operations.len();
        self.paint_operations
            .push(PaintOperation::StartTextShadowGroup {
                bounds,
                content_mask,
                scale_factor,
                shadows: shadows.clone(),
            });
        self.active_text_shadow_group = Some(ActiveTextShadowGroup {
            order,
            shadows,
            scale_factor,
            content_mask,
            scene: Box::default(),
            shadow_scene: Box::default(),
            paint_operation_start,
            color_glyph_attempted: false,
            transformed_glyph_attempted: false,
        });
        Ok(())
    }

    /// Adds an uncolored grayscale glyph to the active shadow source without
    /// changing the separately captured foreground glyph.
    pub(crate) fn insert_text_shadow_glyph(&mut self, sprite: MonochromeSprite) {
        let Some(active) = self.active_text_shadow_group.as_mut() else {
            return;
        };
        if sprite.transformation != TransformationMatrix::unit() {
            active.transformed_glyph_attempted = true;
            return;
        }
        active.shadow_scene.insert_primitive(sprite);
        self.paint_operations
            .push(PaintOperation::TextShadowGlyph(sprite));
    }

    pub(crate) fn has_active_text_shadow_group(&self) -> bool {
        self.active_text_shadow_group.is_some()
    }

    pub(crate) fn mark_color_glyph_in_text_shadow_group(&mut self) {
        if let Some(group) = self.active_text_shadow_group.as_mut() {
            group.color_glyph_attempted = true;
        }
    }

    /// Completes the source alpha group. Invalid groups are removed atomically;
    /// the caller receives an error instead of text painted without shadows.
    pub fn pop_text_shadow_group(&mut self) -> Result<(), TextShadowGroupError> {
        let Some(mut active) = self.active_text_shadow_group.take() else {
            return Err(TextShadowGroupError::NoActiveGroup);
        };
        let validation = if active.color_glyph_attempted {
            Err(TextShadowGroupError::ColorGlyph)
        } else if active.transformed_glyph_attempted {
            Err(TextShadowGroupError::TransformedGlyph)
        } else if !active.scene.layer_stack.is_empty() || active.scene.layer_underflowed {
            Err(TextShadowGroupError::UnbalancedLayer)
        } else if !active.scene.backdrop_blurs.is_empty()
            || !active.scene.quads.is_empty()
            || !active.scene.paths.is_empty()
            || !active.scene.underlines.is_empty()
            || !active.scene.surfaces.is_empty()
            || !active.scene.linear_gradient_mask_groups.is_empty()
            || !active.scene.text_shadow_groups.is_empty()
        {
            Err(TextShadowGroupError::UnsupportedForegroundPrimitive)
        } else if active.shadow_scene.monochrome_sprites.is_empty() {
            Err(TextShadowGroupError::EmptyGlyphSource)
        } else {
            Ok(())
        };
        if let Err(error) = validation {
            self.paint_operations.truncate(active.paint_operation_start);
            return Err(error);
        }

        let mut source_bounds = active.shadow_scene.monochrome_sprites[0].bounds.intersect(
            &active.shadow_scene.monochrome_sprites[0]
                .content_mask
                .bounds,
        );
        for sprite in active.shadow_scene.monochrome_sprites.iter().skip(1) {
            source_bounds =
                source_bounds.union(&sprite.bounds.intersect(&sprite.content_mask.bounds));
        }
        if source_bounds.is_empty() {
            self.paint_operations.truncate(active.paint_operation_start);
            return Err(TextShadowGroupError::EmptyGlyphSource);
        }
        let shadows = match active
            .shadows
            .into_iter()
            .map(|shadow| shadow.scale(active.scale_factor, source_bounds, active.content_mask))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(shadows) => shadows,
            Err(error) => {
                self.paint_operations.truncate(active.paint_operation_start);
                return Err(error);
            }
        };
        active.scene.finish();
        active.shadow_scene.finish();
        self.text_shadow_groups.push(TextShadowGroup {
            order: active.order,
            scene: active.scene,
            shadow_scene: active.shadow_scene,
            shadows,
        });
        self.paint_operations
            .push(PaintOperation::EndTextShadowGroup);
        Ok(())
    }

    /// Finishes the active group, rejecting child constructs that cannot be
    /// sampled by all native renderers. A rejected group's children are
    /// discarded rather than painted without their mask.
    pub fn pop_linear_gradient_mask_group(&mut self) -> Result<(), LinearGradientMaskGroupError> {
        let Some(mut active) = self.active_linear_gradient_mask_group.take() else {
            return Err(LinearGradientMaskGroupError::NoActiveGroup);
        };
        let validation = if !active.scene.layer_stack.is_empty() || active.scene.layer_underflowed {
            Err(LinearGradientMaskGroupError::UnbalancedLayer)
        } else if !active.scene.backdrop_blurs.is_empty() {
            Err(LinearGradientMaskGroupError::BackdropBlur)
        } else if !active.scene.surfaces.is_empty() {
            Err(LinearGradientMaskGroupError::Surface)
        } else if active.color_glyph_attempted {
            Err(LinearGradientMaskGroupError::ColorGlyph)
        } else if !active.scene.subpixel_sprites.is_empty() {
            Err(LinearGradientMaskGroupError::UnconvertedSubpixelText)
        } else {
            Ok(())
        };
        if let Err(error) = validation {
            self.paint_operations.truncate(active.paint_operation_start);
            return Err(error);
        }
        active.scene.finish();
        if active.scene.paint_operations.is_empty() {
            self.paint_operations.truncate(active.paint_operation_start);
            return Ok(());
        }
        self.linear_gradient_mask_groups
            .push(LinearGradientMaskGroup {
                order: active.order,
                mask: active.mask,
                scene: active.scene,
            });
        self.paint_operations
            .push(PaintOperation::EndLinearGradientMaskGroup);
        Ok(())
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::BackdropBlur(blur) => self.insert_backdrop_blur(*blur),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::StartLinearGradientMaskGroup(mask) => {
                    self.push_linear_gradient_mask_group(*mask)
                        .expect("validated mask groups cannot become nested during replay");
                }
                PaintOperation::EndLinearGradientMaskGroup => {
                    self.pop_linear_gradient_mask_group()
                        .expect("validated mask group replay must remain representable");
                }
                PaintOperation::StartTextShadowGroup {
                    bounds,
                    content_mask,
                    scale_factor,
                    shadows,
                } => self
                    .push_text_shadow_group(*bounds, *content_mask, *scale_factor, shadows.clone())
                    .expect("validated text-shadow group replay must remain representable"),
                PaintOperation::TextShadowGlyph(sprite) => self.insert_text_shadow_glyph(*sprite),
                PaintOperation::EndTextShadowGroup => self
                    .pop_text_shadow_group()
                    .expect("validated text-shadow group replay must remain representable"),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.backdrop_blurs.sort_by_key(|blur| blur.order);
        self.linear_gradient_mask_groups
            .sort_by_key(|group| group.order);
        self.text_shadow_groups.sort_by_key(|group| group.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            linear_gradient_mask_groups_start: 0,
            linear_gradient_mask_groups_iter: self.linear_gradient_mask_groups.iter().peekable(),
            text_shadow_groups_start: 0,
            text_shadow_groups_iter: self.text_shadow_groups.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    TextShadowGroup,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    LinearGradientMaskGroup,
}

#[derive(Clone)]
pub(crate) enum PaintOperation {
    Primitive(Primitive),
    BackdropBlur(BackdropBlur),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    StartLinearGradientMaskGroup(LinearGradientMaskParams),
    EndLinearGradientMaskGroup,
    StartTextShadowGroup {
        bounds: Bounds<ScaledPixels>,
        content_mask: ContentMask<ScaledPixels>,
        scale_factor: f32,
        shadows: Vec<TextShadow>,
    },
    TextShadowGlyph(MonochromeSprite),
    EndTextShadowGroup,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    linear_gradient_mask_groups_start: usize,
    linear_gradient_mask_groups_iter: Peekable<slice::Iter<'a, LinearGradientMaskGroup>>,
    text_shadow_groups_start: usize,
    text_shadow_groups_iter: Peekable<slice::Iter<'a, TextShadowGroup>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.text_shadow_groups_iter.peek().map(|g| g.order),
                PrimitiveKind::TextShadowGroup,
            ),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.linear_gradient_mask_groups_iter
                    .peek()
                    .map(|g| g.order),
                PrimitiveKind::LinearGradientMaskGroup,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::TextShadowGroup => {
                let group_index = self.text_shadow_groups_start;
                self.text_shadow_groups_iter.next();
                self.text_shadow_groups_start += 1;
                Some(PrimitiveBatch::TextShadowGroup(group_index))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::LinearGradientMaskGroup => {
                let group_index = self.linear_gradient_mask_groups_start;
                self.linear_gradient_mask_groups_iter.next();
                self.linear_gradient_mask_groups_start += 1;
                Some(PrimitiveBatch::LinearGradientMaskGroup(group_index))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    TextShadowGroup(usize),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    LinearGradientMaskGroup(usize),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::TextShadowGroup(_) => "text shadow group".into(),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::LinearGradientMaskGroup(_) => "linear gradient mask group".into(),
        }
    }
}

/// Per-primitive scoped edge fade (see `Window::with_edge_fade`): the
/// fragment shader multiplies alpha by a squared ramp measured from these
/// window-space edges (device pixels) — a TRUE per-pixel fade, so large
/// fills and images dissolve across the band instead of popping at their
/// bounding-box edge. A zero band disables that edge; zeroed = no fade.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct EdgeFadeParams {
    pub top_y: f32,
    pub bottom_y: f32,
    pub band_top: f32,
    pub band_bottom: f32,
    pub left_x: f32,
    pub right_x: f32,
    pub band_left: f32,
    pub band_right: f32,
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_colors: Edges<Hsla>,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub fade: EdgeFadeParams,
    pub mask_alignment_pad: u32,
    pub mask: LinearGradientMaskParams,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

/// A within-window backdrop blur region: the renderer snapshots everything
/// painted below this order and paints it back gaussian-blurred inside the
/// rounded bounds (frosted-glass popovers). macOS Metal only — see
/// [`crate::Window::paint_backdrop_blur`].
#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct BackdropBlur {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub fade: EdgeFadeParams,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod linear_gradient_mask_tests {
    use super::*;

    fn mask_bounds() -> Bounds<Pixels> {
        Bounds {
            origin: point(Pixels(10.0), Pixels(20.0)),
            size: Size {
                width: Pixels(200.0),
                height: Pixels(100.0),
            },
        }
    }

    fn stop(alpha: f32, percentage: f32, offset: f32) -> LinearGradientMaskStop {
        LinearGradientMaskStop {
            alpha,
            percentage,
            offset: Pixels(offset),
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON * 8.0,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn locked_vertical_ascii_fade_matches_source_stops() {
        let mask = LinearGradientMask::try_new(
            mask_bounds(),
            LinearGradientMaskDirection::ToBottom,
            &[
                stop(1.0, 0.0, 0.0),
                stop(1.0, 0.7, 0.0),
                stop(0.0, 1.0, 0.0),
            ],
        )
        .unwrap();

        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(20.0))), 1.0);
        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(90.0))), 1.0);
        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(105.0))), 0.5);
        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(120.0))), 0.0);
    }

    #[test]
    fn locked_ticker_mask_resolves_pixel_and_calc_stops() {
        let mask = LinearGradientMask::try_new(
            mask_bounds(),
            LinearGradientMaskDirection::ToRight,
            &[
                stop(0.0, 0.0, 0.0),
                stop(1.0, 0.0, 30.0),
                stop(1.0, 1.0, -30.0),
                stop(0.0, 1.0, 0.0),
            ],
        )
        .unwrap();

        assert_close(mask.alpha_at(point(Pixels(10.0), Pixels(50.0))), 0.0);
        assert_close(mask.alpha_at(point(Pixels(25.0), Pixels(50.0))), 0.5);
        assert_close(mask.alpha_at(point(Pixels(110.0), Pixels(50.0))), 1.0);
        assert_close(mask.alpha_at(point(Pixels(195.0), Pixels(50.0))), 0.5);
        assert_close(mask.alpha_at(point(Pixels(210.0), Pixels(50.0))), 0.0);
    }

    #[test]
    fn locked_soft_skin_mask_preserves_bottom_to_top_alpha() {
        let mask = LinearGradientMask::try_new(
            mask_bounds(),
            LinearGradientMaskDirection::ToTop,
            &[
                stop(1.0, 0.06, 0.0),
                stop(0.4, 0.52, 0.0),
                stop(0.0, 0.9, 0.0),
            ],
        )
        .unwrap();

        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(120.0))), 1.0);
        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(68.0))), 0.4);
        assert_close(mask.alpha_at(point(Pixels(50.0), Pixels(30.0))), 0.0);
    }

    #[test]
    fn invalid_masks_fail_closed() {
        assert_eq!(
            LinearGradientMask::try_new(
                mask_bounds(),
                LinearGradientMaskDirection::ToBottom,
                &[stop(1.0, 0.0, 0.0)]
            ),
            Err(LinearGradientMaskError::InvalidStopCount(1))
        );
        assert_eq!(
            LinearGradientMask::try_new(
                mask_bounds(),
                LinearGradientMaskDirection::ToBottom,
                &[stop(1.0, 0.8, 0.0), stop(0.0, 0.2, 0.0)]
            ),
            Err(LinearGradientMaskError::StopsOutOfOrder(1))
        );
        assert_eq!(
            LinearGradientMask::try_new(
                mask_bounds(),
                LinearGradientMaskDirection::ToBottom,
                &[stop(1.0, 0.0, 0.0), stop(f32::NAN, 1.0, 0.0)]
            ),
            Err(LinearGradientMaskError::InvalidStop(1))
        );
        assert_eq!(
            LinearGradientMask::try_new(
                Bounds::default(),
                LinearGradientMaskDirection::ToBottom,
                &[stop(1.0, 0.0, 0.0), stop(0.0, 1.0, 0.0)]
            ),
            Err(LinearGradientMaskError::InvalidBounds)
        );
    }

    #[test]
    fn mask_shader_abi_has_explicit_cross_renderer_stride() {
        assert_eq!(std::mem::size_of::<LinearGradientMaskStop>(), 12);
        assert_eq!(std::mem::size_of::<ScaledLinearGradientMaskStop>(), 12);
        assert_eq!(std::mem::size_of::<LinearGradientMaskParams>(), 88);
        assert_eq!(std::mem::align_of::<LinearGradientMaskParams>(), 8);
        assert_eq!(
            std::mem::offset_of!(Quad, mask),
            std::mem::offset_of!(Quad, fade) + std::mem::size_of::<EdgeFadeParams>() + 4
        );
        assert_eq!(std::mem::offset_of!(Quad, mask) % 8, 0);
    }

    #[test]
    fn no_mask_params_are_an_alpha_identity() {
        let params = LinearGradientMaskParams::default();
        assert_eq!(params.stop_count, 0);
        let mask = LinearGradientMask::try_new(
            mask_bounds(),
            LinearGradientMaskDirection::ToLeft,
            &[stop(0.0, 0.0, 0.0), stop(1.0, 1.0, 0.0)],
        )
        .unwrap();
        assert_close(mask.alpha_at(point(Pixels(210.0), Pixels(50.0))), 0.0);
        assert_close(mask.alpha_at(point(Pixels(10.0), Pixels(50.0))), 1.0);
    }

    #[test]
    fn paint_quad_retains_and_scales_the_validated_mask() {
        let mask = LinearGradientMask::try_new(
            mask_bounds(),
            LinearGradientMaskDirection::ToRight,
            &[stop(0.0, 0.0, 30.0), stop(1.0, 1.0, -30.0)],
        )
        .unwrap();
        let paint_quad = crate::fill(mask_bounds(), Hsla::default()).linear_gradient_mask(mask);
        assert_eq!(paint_quad.mask, Some(mask));

        let scaled = mask.scale(2.0);
        assert_eq!(
            scaled.bounds.origin,
            point(ScaledPixels(20.0), ScaledPixels(40.0))
        );
        assert_eq!(
            scaled.bounds.size,
            Size {
                width: ScaledPixels(400.0),
                height: ScaledPixels(200.0),
            }
        );
        assert_eq!(scaled.stops[0].offset, ScaledPixels(60.0));
        assert_eq!(scaled.stops[1].offset, ScaledPixels(-60.0));
    }

    #[test]
    fn every_renderer_applies_the_native_mask_to_quad_fill_and_border() {
        for (renderer, shader) in [
            (
                "DirectX HLSL",
                include_str!("../../gpui_windows/src/shaders.hlsl"),
            ),
            (
                "WGPU WGSL",
                include_str!("../../gpui_wgpu/src/shaders.wgsl"),
            ),
            (
                "macOS Metal",
                include_str!("../../gpui_macos/src/shaders.metal"),
            ),
        ] {
            assert!(
                shader.contains("linear_gradient_mask_alpha"),
                "{renderer} is missing the native mask evaluator"
            );
            assert!(
                shader.contains("background_color.a *= mask_alpha"),
                "{renderer} does not mask quad fills"
            );
            assert!(
                shader.contains("border_color.a *= mask_alpha"),
                "{renderer} does not mask quad borders"
            );
            assert!(
                shader.contains("stop_count")
                    && shader.contains("percentage")
                    && shader.contains("offset"),
                "{renderer} is missing the typed percentage-plus-pixel stop ABI"
            );
        }
    }

    fn scaled_locked_mask() -> LinearGradientMaskParams {
        LinearGradientMask::try_new(
            mask_bounds(),
            LinearGradientMaskDirection::ToBottom,
            &[stop(1.0, 0.0, 0.0), stop(0.0, 1.0, 0.0)],
        )
        .unwrap()
        .scale(1.0)
    }

    fn group_quad(x: f32) -> Quad {
        let bounds = Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(20.0)),
            size: Size {
                width: ScaledPixels(80.0),
                height: ScaledPixels(80.0),
            },
        };
        Quad {
            bounds,
            content_mask: ContentMask { bounds },
            ..Default::default()
        }
    }

    fn group_atlas_tile(kind: crate::AtlasTextureKind) -> AtlasTile {
        AtlasTile {
            texture_id: AtlasTextureId { index: 7, kind },
            tile_id: crate::TileId(11),
            padding: 0,
            bounds: Bounds {
                origin: Point::default(),
                size: Size {
                    width: crate::DevicePixels(8),
                    height: crate::DevicePixels(12),
                },
            },
        }
    }

    fn group_monochrome_sprite() -> MonochromeSprite {
        let bounds = group_quad(10.0).bounds;
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: ContentMask { bounds },
            color: Hsla::default(),
            tile: group_atlas_tile(crate::AtlasTextureKind::Monochrome),
            transformation: TransformationMatrix::unit(),
        }
    }

    fn group_subpixel_sprite() -> SubpixelSprite {
        let mono = group_monochrome_sprite();
        SubpixelSprite {
            order: mono.order,
            pad: mono.pad,
            bounds: mono.bounds,
            content_mask: mono.content_mask,
            color: mono.color,
            tile: group_atlas_tile(crate::AtlasTextureKind::Subpixel),
            transformation: mono.transformation,
        }
    }

    #[test]
    fn mask_group_preserves_overlapping_children_as_one_atomic_batch() {
        let mut scene = Scene::default();
        scene
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        scene.insert_primitive(group_quad(10.0));
        scene.insert_primitive(group_quad(40.0));
        scene.pop_linear_gradient_mask_group().unwrap();
        scene.finish();

        assert!(
            scene.quads.is_empty(),
            "children must not leak into the parent"
        );
        assert_eq!(scene.linear_gradient_mask_groups.len(), 1);
        assert_eq!(scene.linear_gradient_mask_groups[0].scene.quads.len(), 2);
        assert!(matches!(
            scene.batches().collect::<Vec<_>>().as_slice(),
            [PrimitiveBatch::LinearGradientMaskGroup(0)]
        ));
    }

    #[test]
    fn grayscale_glyph_sprite_is_flattened_inside_the_mask_group() {
        let mut scene = Scene::default();
        scene
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        scene.insert_primitive(group_monochrome_sprite());
        scene.pop_linear_gradient_mask_group().unwrap();
        scene.finish();

        let group = &scene.linear_gradient_mask_groups[0].scene;
        assert_eq!(group.monochrome_sprites.len(), 1);
        assert!(group.subpixel_sprites.is_empty());
        assert!(matches!(
            group.batches().collect::<Vec<_>>().as_slice(),
            [PrimitiveBatch::MonochromeSprites { .. }]
        ));
    }

    #[test]
    fn stale_lcd_sprite_and_color_glyph_attempts_fail_closed() {
        let mut stale_lcd = Scene::default();
        stale_lcd
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        stale_lcd.insert_primitive(group_subpixel_sprite());
        assert_eq!(
            stale_lcd.pop_linear_gradient_mask_group(),
            Err(LinearGradientMaskGroupError::UnconvertedSubpixelText)
        );
        assert!(stale_lcd.linear_gradient_mask_groups.is_empty());
        assert_eq!(stale_lcd.len(), 0);

        let mut color_glyph = Scene::default();
        color_glyph
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        color_glyph.mark_color_glyph_in_linear_gradient_mask_group();
        assert_eq!(
            color_glyph.pop_linear_gradient_mask_group(),
            Err(LinearGradientMaskGroupError::ColorGlyph)
        );
        assert!(color_glyph.linear_gradient_mask_groups.is_empty());
        assert_eq!(color_glyph.len(), 0);
    }

    #[test]
    fn group_masking_differs_from_incorrect_per_primitive_masking_at_overlap() {
        let child_alpha = 0.5_f32;
        let mask_alpha = 0.5_f32;
        let flattened_alpha = child_alpha + child_alpha * (1.0 - child_alpha);
        let exact_group_alpha = flattened_alpha * mask_alpha;
        let individually_masked_alpha =
            child_alpha * mask_alpha + child_alpha * mask_alpha * (1.0 - child_alpha * mask_alpha);

        assert_close(exact_group_alpha, 0.375);
        assert_close(individually_masked_alpha, 0.4375);
        assert_ne!(exact_group_alpha, individually_masked_alpha);
    }

    #[test]
    fn mask_group_replays_as_an_atomic_subscene() {
        let mut previous = Scene::default();
        previous
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        previous.insert_primitive(group_quad(10.0));
        previous.insert_primitive(group_quad(40.0));
        previous.pop_linear_gradient_mask_group().unwrap();

        let mut replayed = Scene::default();
        replayed.replay(0..previous.len(), &previous);
        replayed.finish();
        assert_eq!(replayed.linear_gradient_mask_groups.len(), 1);
        assert_eq!(replayed.linear_gradient_mask_groups[0].scene.quads.len(), 2);
    }

    #[test]
    fn unsupported_mask_group_children_fail_closed_without_leaking() {
        let mut scene = Scene::default();
        scene
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        scene.insert_primitive(group_quad(10.0));
        scene.insert_backdrop_blur(BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(4.0),
            bounds: scaled_locked_mask().bounds,
            content_mask: ContentMask {
                bounds: scaled_locked_mask().bounds,
            },
            corner_radii: Default::default(),
        });
        assert_eq!(
            scene.pop_linear_gradient_mask_group(),
            Err(LinearGradientMaskGroupError::BackdropBlur)
        );
        assert!(scene.quads.is_empty());
        assert!(scene.linear_gradient_mask_groups.is_empty());
        assert_eq!(scene.len(), 0);
    }

    #[test]
    fn nested_mask_groups_fail_before_accepting_children() {
        let mut scene = Scene::default();
        scene
            .push_linear_gradient_mask_group(scaled_locked_mask())
            .unwrap();
        assert_eq!(
            scene.push_linear_gradient_mask_group(scaled_locked_mask()),
            Err(LinearGradientMaskGroupError::NestedGroup)
        );
    }

    #[test]
    fn every_renderer_flattens_then_masks_the_group_once() {
        for (renderer, shader) in [
            (
                "DirectX HLSL",
                include_str!("../../gpui_windows/src/shaders.hlsl"),
            ),
            (
                "WGPU WGSL",
                include_str!("../../gpui_wgpu/src/shaders.wgsl"),
            ),
            (
                "macOS Metal",
                include_str!("../../gpui_macos/src/shaders.metal"),
            ),
        ] {
            assert!(
                shader.contains("linear_gradient_mask_group_vertex")
                    && shader.contains("linear_gradient_mask_group_fragment"),
                "{renderer} is missing the group composite shader pair"
            );
            assert!(
                shader.contains("flattened_group") && shader.contains("* mask_alpha"),
                "{renderer} does not apply the mask once to the flattened group texture"
            );
        }
    }

    #[test]
    fn chromium_masked_glyph_alpha_oracle_matches_group_composite_math() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/masked_text_alpha_chromium.json"
        ))
        .unwrap();
        assert_eq!(oracle["chromiumVersion"], "151.0.7922.170");
        assert_eq!(oracle["deviceScaleFactor"], 1);
        assert_eq!(oracle["unmaskedNonzeroPixels"], 519);
        assert_eq!(oracle["maskedNonzeroPixels"], 519);
        assert_eq!(
            oracle["pngSha256"],
            "A1E5FE02DEBB5F913D0545C37C3AED4A489BC9572DB4EF2FAC3928BB6026E468"
        );

        let width = oracle["localMaskWidth"].as_f64().unwrap();
        for sample in oracle["samples"].as_array().unwrap() {
            let x = sample[0].as_f64().unwrap();
            let unmasked = sample[2].as_f64().unwrap();
            let masked = sample[3].as_f64().unwrap();
            // Chromium samples the generated gradient at the pixel center,
            // then multiplies the flattened glyph alpha by that mask alpha.
            let expected = (unmasked * ((x + 0.5) / width)).round();
            assert!(
                (masked - expected).abs() <= 1.0,
                "x={x}: expected {expected}, got {masked}"
            );
        }
    }

    #[test]
    fn every_group_backend_uses_the_grayscale_sprite_path() {
        let directx = include_str!("../../gpui_windows/src/directx_renderer.rs");
        assert!(directx.contains("self.draw_scene_batches(&group.scene"));
        assert!(directx.contains("PrimitiveBatch::MonochromeSprites"));

        let wgpu = include_str!("../../gpui_wgpu/src/wgpu_renderer.rs");
        assert!(wgpu.contains("fn encode_flat_scene("));
        assert!(wgpu.contains("PrimitiveBatch::MonochromeSprites"));

        let metal = include_str!("../../gpui_macos/src/metal_renderer.rs");
        assert!(metal.contains("fn draw_flat_scene_to_texture("));
        assert!(metal.contains("PrimitiveBatch::MonochromeSprites"));
        assert!(metal.contains("PrimitiveBatch::SubpixelSprites { .. }"));

        for (renderer, shader, alpha_expression, composite_expression) in [
            (
                "DirectX HLSL",
                include_str!("../../gpui_windows/src/shaders.hlsl"),
                "input.color.a * alpha_corrected",
                "flattened_group * mask_alpha",
            ),
            (
                "WGPU WGSL",
                include_str!("../../gpui_wgpu/src/shaders.wgsl"),
                "blend_color(input.color, alpha_corrected)",
                "flattened_group * mask_alpha",
            ),
            (
                "macOS Metal",
                include_str!("../../gpui_macos/src/shaders.metal"),
                "color.a *= sample.a",
                "flattened_group_pixel * mask_alpha",
            ),
        ] {
            assert!(
                shader.contains(alpha_expression),
                "{renderer} does not preserve grayscale glyph alpha before group compositing"
            );
            assert!(
                shader.contains(composite_expression),
                "{renderer} does not multiply the flattened glyph alpha by the mask"
            );
        }
    }
}

#[cfg(test)]
mod text_shadow_tests {
    use super::*;

    const ORACLE_WIDTH: usize = 128;
    const ORACLE_HEIGHT: usize = 96;
    const ORACLE_PLANE_LEN: usize = ORACLE_WIDTH * ORACLE_HEIGHT;

    fn shadow_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(50.0), ScaledPixels(29.0)),
            size: Size {
                width: ScaledPixels(28.0),
                height: ScaledPixels(35.0),
            },
        }
    }

    fn shadow(blur_radius: f32, offset_y: f32) -> TextShadow {
        TextShadow {
            offset: point(Pixels(0.0), Pixels(offset_y)),
            blur_radius: Pixels(blur_radius),
            color: Hsla::default(),
        }
    }

    fn atlas_tile(kind: crate::AtlasTextureKind) -> AtlasTile {
        AtlasTile {
            texture_id: AtlasTextureId { index: 3, kind },
            tile_id: crate::TileId(5),
            padding: 0,
            bounds: Bounds {
                origin: Point::default(),
                size: Size {
                    width: crate::DevicePixels(28),
                    height: crate::DevicePixels(35),
                },
            },
        }
    }

    fn monochrome_sprite(transformation: TransformationMatrix) -> MonochromeSprite {
        let bounds = shadow_bounds();
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds,
            content_mask: ContentMask { bounds },
            color: Hsla::default(),
            tile: atlas_tile(crate::AtlasTextureKind::Monochrome),
            transformation,
        }
    }

    fn pass_a8(source: &[u8], kernel: &TextShadowKernel, vertical: bool, small: bool) -> Vec<u8> {
        let mut output = vec![0; source.len()];
        for y in 0..ORACLE_HEIGHT {
            for x in 0..ORACLE_WIDTH {
                if small {
                    let mut sum = 128_u32;
                    for (distance, weight) in kernel.values.iter().enumerate() {
                        let coefficient = (weight * 65_536.0).round() as u32;
                        for sign in if distance == 0 { 0..=0 } else { -1..=1 } {
                            if distance != 0 && sign == 0 {
                                continue;
                            }
                            let sample_x = x as isize
                                + if vertical {
                                    0
                                } else {
                                    sign * distance as isize
                                };
                            let sample_y = y as isize
                                + if vertical {
                                    sign * distance as isize
                                } else {
                                    0
                                };
                            if (0..ORACLE_WIDTH as isize).contains(&sample_x)
                                && (0..ORACLE_HEIGHT as isize).contains(&sample_y)
                            {
                                let alpha = source
                                    [sample_y as usize * ORACLE_WIDTH + sample_x as usize]
                                    as u32;
                                sum += (alpha * 256 * coefficient) >> 16;
                            }
                        }
                    }
                    output[y * ORACLE_WIDTH + x] = (sum >> 8) as u8;
                } else {
                    let mut sum = 0_u32;
                    for (distance, coefficient) in kernel.values.iter().enumerate() {
                        for sign in if distance == 0 { 0..=0 } else { -1..=1 } {
                            if distance != 0 && sign == 0 {
                                continue;
                            }
                            let sample_x = x as isize
                                + if vertical {
                                    0
                                } else {
                                    sign * distance as isize
                                };
                            let sample_y = y as isize
                                + if vertical {
                                    sign * distance as isize
                                } else {
                                    0
                                };
                            if (0..ORACLE_WIDTH as isize).contains(&sample_x)
                                && (0..ORACLE_HEIGHT as isize).contains(&sample_y)
                            {
                                sum += u32::from(
                                    source[sample_y as usize * ORACLE_WIDTH + sample_x as usize],
                                ) * coefficient.round() as u32;
                            }
                        }
                    }
                    output[y * ORACLE_WIDTH + x] = (((u64::from(sum)
                        * u64::from(kernel.normalization_weight))
                        + (1_u64 << 31))
                        >> 32) as u8;
                }
            }
        }
        output
    }

    fn blur_a8(source: &[u8], blur_radius: f32) -> Vec<u8> {
        let kernel = text_shadow_kernel(blur_radius).unwrap();
        if blur_radius < 4.0 {
            let vertical = pass_a8(source, &kernel, true, true);
            pass_a8(&vertical, &kernel, false, true)
        } else {
            let horizontal = pass_a8(source, &kernel, false, false);
            pass_a8(&horizontal, &kernel, true, false)
        }
    }

    #[test]
    fn locked_blur_kernels_match_every_chromium_a8_pixel() {
        let metadata: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/text_shadow_alpha_chromium.json"
        ))
        .unwrap();
        assert_eq!(metadata["chromiumVersion"], "151.0.7922.170");
        assert_eq!(metadata["deviceScaleFactor"], 1);
        assert_eq!(metadata["lockedDeclarations"].as_array().unwrap().len(), 5);
        assert_eq!(metadata["lockedBackgrounds"].as_array().unwrap().len(), 6);

        let planes = include_bytes!("../tests/fixtures/text_shadow_alpha_chromium.a8");
        assert_eq!(planes.len(), ORACLE_PLANE_LEN * 5);
        let source = &planes[..ORACLE_PLANE_LEN];
        for (blur_radius, target_index) in [(3.0, 1), (4.0, 2), (9.0, 3)] {
            let actual = blur_a8(source, blur_radius);
            let expected =
                &planes[target_index * ORACLE_PLANE_LEN..(target_index + 1) * ORACLE_PLANE_LEN];
            assert_eq!(actual, expected, "Chromium A8 drift for {blur_radius}px");
        }
    }

    #[test]
    fn locked_kernel_shapes_and_gpu_abi_are_stable() {
        assert_eq!(text_shadow_kernel(3.0).unwrap().values.len(), 4);
        assert_eq!(text_shadow_kernel(4.0).unwrap().values.len(), 6);
        assert_eq!(text_shadow_kernel(9.0).unwrap().values.len(), 12);
        assert_eq!(std::mem::size_of::<ScaledTextShadow>(), 360);
        assert_eq!(std::mem::align_of::<ScaledTextShadow>(), 8);

        let small = shadow(3.0, 0.0)
            .scale(
                1.0,
                shadow_bounds(),
                ContentMask {
                    bounds: shadow_bounds(),
                },
            )
            .unwrap();
        assert_eq!(small.first_pass_bounds.origin.x, shadow_bounds().origin.x);
        assert!(small.first_pass_bounds.origin.y < shadow_bounds().origin.y);

        let large = shadow(4.0, 0.0)
            .scale(
                1.0,
                shadow_bounds(),
                ContentMask {
                    bounds: shadow_bounds(),
                },
            )
            .unwrap();
        assert!(large.first_pass_bounds.origin.x < shadow_bounds().origin.x);
        assert_eq!(large.first_pass_bounds.origin.y, shadow_bounds().origin.y);
    }

    #[test]
    fn group_captures_foreground_and_independent_shadow_alpha_atomically() {
        let mut scene = Scene::default();
        scene
            .push_text_shadow_group(
                shadow_bounds(),
                ContentMask {
                    bounds: shadow_bounds(),
                },
                1.0,
                vec![shadow(4.0, 0.0), shadow(9.0, 0.0), shadow(3.0, 1.0)],
            )
            .unwrap();
        let foreground = monochrome_sprite(TransformationMatrix::unit());
        scene.insert_primitive(foreground);
        scene.insert_text_shadow_glyph(MonochromeSprite {
            color: Hsla {
                l: 1.0,
                a: 1.0,
                ..Hsla::default()
            },
            ..foreground
        });
        scene.pop_text_shadow_group().unwrap();
        scene.finish();

        assert!(scene.monochrome_sprites.is_empty());
        assert_eq!(scene.text_shadow_groups.len(), 1);
        let group = &scene.text_shadow_groups[0];
        assert_eq!(group.scene.monochrome_sprites.len(), 1);
        assert_eq!(group.shadow_scene.monochrome_sprites.len(), 1);
        assert_eq!(group.shadows.len(), 3);
        assert!(matches!(
            scene.batches().collect::<Vec<_>>().as_slice(),
            [PrimitiveBatch::TextShadowGroup(0)]
        ));
    }

    #[test]
    fn unsupported_text_shadow_cases_fail_closed() {
        let content_mask = ContentMask {
            bounds: shadow_bounds(),
        };
        assert_eq!(
            shadow(3.0, 1.0).scale(1.25, shadow_bounds(), content_mask),
            Err(TextShadowGroupError::FractionalDeviceOffset)
        );
        assert_eq!(
            shadow(-1.0, 0.0).scale(1.0, shadow_bounds(), content_mask),
            Err(TextShadowGroupError::NegativeBlurRadius)
        );

        let mut transformed = Scene::default();
        transformed
            .push_text_shadow_group(shadow_bounds(), content_mask, 1.0, vec![shadow(3.0, 0.0)])
            .unwrap();
        transformed.insert_primitive(monochrome_sprite(TransformationMatrix::unit()));
        let mut matrix = TransformationMatrix::unit();
        matrix.translation[0] = 1.0;
        transformed.insert_text_shadow_glyph(monochrome_sprite(matrix));
        assert_eq!(
            transformed.pop_text_shadow_group(),
            Err(TextShadowGroupError::TransformedGlyph)
        );
        assert_eq!(transformed.len(), 0);

        let mut color = Scene::default();
        color
            .push_text_shadow_group(shadow_bounds(), content_mask, 1.0, vec![shadow(3.0, 0.0)])
            .unwrap();
        color.mark_color_glyph_in_text_shadow_group();
        assert_eq!(
            color.pop_text_shadow_group(),
            Err(TextShadowGroupError::ColorGlyph)
        );
        assert_eq!(color.len(), 0);
    }

    #[test]
    fn every_renderer_uses_the_two_pass_a8_shadow_contract() {
        for (renderer, shader) in [
            (
                "DirectX HLSL",
                include_str!("../../gpui_windows/src/shaders.hlsl"),
            ),
            (
                "WGPU WGSL",
                include_str!("../../gpui_wgpu/src/shaders.wgsl"),
            ),
            (
                "macOS Metal",
                include_str!("../../gpui_macos/src/shaders.metal"),
            ),
        ] {
            assert!(
                shader.contains("text_shadow_blur_vertex")
                    && shader.contains("text_shadow_composite_fragment"),
                "{renderer} is missing the two-pass shader pair"
            );
            assert!(
                shader.contains("source_a8")
                    && shader.contains("256.0 * weight")
                    && shader.contains("floor(alpha / 256.0)"),
                "{renderer} does not reproduce Skia's small-blur 8.8/0.16 arithmetic"
            );
            assert!(
                shader.contains("shadow.blur_radius < 4.0"),
                "{renderer} does not preserve the observable Skia pass-order boundary"
            );
        }

        for renderer in [
            include_str!("../../gpui_windows/src/directx_renderer.rs"),
            include_str!("../../gpui_wgpu/src/wgpu_renderer.rs"),
            include_str!("../../gpui_macos/src/metal_renderer.rs"),
        ] {
            assert!(renderer.contains("group.shadows.iter().rev()"));
            assert!(renderer.contains("group.shadow_scene"));
            assert!(renderer.contains("group.scene"));
        }
    }
}

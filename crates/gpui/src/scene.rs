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

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
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
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.backdrop_blurs.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_backdrop_blur(&mut self, mut blur: BackdropBlur) {
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

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::BackdropBlur(blur) => self.insert_backdrop_blur(*blur),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
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
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    BackdropBlur(BackdropBlur),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
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
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
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
}

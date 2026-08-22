use crate::{FontStyle, SharedString};
use std::{borrow::Cow, collections::BTreeSet, ops::RangeInclusive, sync::Arc};
use thiserror::Error;

/// An inclusive Unicode scalar range from a CSS `unicode-range` descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CssUnicodeRange {
    start: u32,
    end: u32,
}

impl CssUnicodeRange {
    /// Construct a validated inclusive Unicode range.
    pub fn new(start: u32, end: u32) -> Result<Self, CssFontFaceRegistryError> {
        if start > end || end > char::MAX as u32 {
            return Err(CssFontFaceRegistryError::InvalidUnicodeRange { start, end });
        }
        Ok(Self { start, end })
    }

    /// The first code point in the inclusive range.
    pub fn start(self) -> u32 {
        self.start
    }

    /// The last code point in the inclusive range.
    pub fn end(self) -> u32 {
        self.end
    }

    /// Whether this range contains a Unicode scalar.
    pub fn contains(self, character: char) -> bool {
        (self.start..=self.end).contains(&(character as u32))
    }
}

/// One byte-exact embedded font response body.
#[derive(Clone, Debug)]
pub struct EmbeddedFontResource {
    id: SharedString,
    bytes: Cow<'static, [u8]>,
}

impl EmbeddedFontResource {
    /// Construct an embedded font body. Registry validation rejects empty or
    /// duplicate ids and empty bodies.
    pub fn new(id: impl Into<SharedString>, bytes: Cow<'static, [u8]>) -> Self {
        Self {
            id: id.into(),
            bytes,
        }
    }

    /// The stable source/resource identity.
    pub fn id(&self) -> &SharedString {
        &self.id
    }

    /// The exact font response bytes.
    pub fn bytes(&self) -> &Cow<'static, [u8]> {
        &self.bytes
    }

    /// A deterministic internal family alias used to select this body without
    /// conflating it with another same-family Unicode subset.
    pub fn native_family_alias(&self) -> SharedString {
        format!(".GPUIEmbeddedFont.{}", self.id).into()
    }
}

/// A typed CSS `@font-face` rule referencing one embedded response body.
#[derive(Clone, Debug)]
pub struct CssFontFace {
    family: SharedString,
    style: FontStyle,
    weights: RangeInclusive<u16>,
    unicode_ranges: Arc<[CssUnicodeRange]>,
    resource_id: SharedString,
}

impl CssFontFace {
    /// Construct a face rule. Full cross-rule validation occurs when the
    /// [`CssFontFaceRegistry`] is built.
    pub fn new(
        family: impl Into<SharedString>,
        style: FontStyle,
        weights: RangeInclusive<u16>,
        unicode_ranges: impl Into<Arc<[CssUnicodeRange]>>,
        resource_id: impl Into<SharedString>,
    ) -> Self {
        Self {
            family: family.into(),
            style,
            weights,
            unicode_ranges: unicode_ranges.into(),
            resource_id: resource_id.into(),
        }
    }

    /// Public CSS family name.
    pub fn family(&self) -> &SharedString {
        &self.family
    }

    /// CSS font style.
    pub fn style(&self) -> FontStyle {
        self.style
    }

    /// Inclusive CSS weight descriptor.
    pub fn weights(&self) -> &RangeInclusive<u16> {
        &self.weights
    }

    /// CSS Unicode coverage descriptors.
    pub fn unicode_ranges(&self) -> &[CssUnicodeRange] {
        &self.unicode_ranges
    }

    /// Referenced embedded resource id.
    pub fn resource_id(&self) -> &SharedString {
        &self.resource_id
    }

    /// Whether this face explicitly covers a Unicode scalar.
    pub fn covers(&self, character: char) -> bool {
        self.unicode_ranges
            .iter()
            .any(|range| range.contains(character))
    }
}

/// A fully validated set of embedded resources and ordered CSS face rules.
///
/// Face order is source order. When otherwise-equivalent rules overlap, the
/// later rule wins, matching CSS cascade order for `@font-face` sources.
#[derive(Clone, Debug)]
pub struct CssFontFaceRegistry {
    resources: Arc<[EmbeddedFontResource]>,
    faces: Arc<[CssFontFace]>,
}

impl CssFontFaceRegistry {
    /// Validate and construct a registry. Unsupported or ambiguous inputs are
    /// rejected before any platform font collection is mutated.
    pub fn new(
        resources: Vec<EmbeddedFontResource>,
        faces: Vec<CssFontFace>,
    ) -> Result<Self, CssFontFaceRegistryError> {
        if resources.is_empty() {
            return Err(CssFontFaceRegistryError::NoResources);
        }
        if faces.is_empty() {
            return Err(CssFontFaceRegistryError::NoFaces);
        }

        let mut resource_ids = BTreeSet::new();
        for resource in &resources {
            if resource.id.is_empty()
                || !resource
                    .id
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
            {
                return Err(CssFontFaceRegistryError::InvalidResourceId(
                    resource.id.to_string(),
                ));
            }
            if resource.bytes.is_empty() {
                return Err(CssFontFaceRegistryError::EmptyResource(
                    resource.id.to_string(),
                ));
            }
            if !resource_ids.insert(resource.id.to_string()) {
                return Err(CssFontFaceRegistryError::DuplicateResource(
                    resource.id.to_string(),
                ));
            }
        }

        let mut used_resources = BTreeSet::new();
        for face in &faces {
            if face.family.trim().is_empty() {
                return Err(CssFontFaceRegistryError::EmptyFamily);
            }
            let (&start, &end) = (face.weights.start(), face.weights.end());
            if start == 0 || start > end || end > 1000 {
                return Err(CssFontFaceRegistryError::InvalidWeightRange { start, end });
            }
            if face.unicode_ranges.is_empty() {
                return Err(CssFontFaceRegistryError::NoUnicodeRanges {
                    family: face.family.to_string(),
                });
            }
            if !resource_ids.contains(face.resource_id.as_ref()) {
                return Err(CssFontFaceRegistryError::UnknownResource(
                    face.resource_id.to_string(),
                ));
            }
            used_resources.insert(face.resource_id.to_string());
        }
        if let Some(unused) = resource_ids.difference(&used_resources).next() {
            return Err(CssFontFaceRegistryError::UnusedResource(unused.clone()));
        }

        Ok(Self {
            resources: resources.into(),
            faces: faces.into(),
        })
    }

    /// Exact embedded resources, in registration order.
    pub fn resources(&self) -> &[EmbeddedFontResource] {
        &self.resources
    }

    /// Ordered CSS face rules.
    pub fn faces(&self) -> &[CssFontFace] {
        &self.faces
    }
}

/// Validation failure for an embedded CSS face registry.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CssFontFaceRegistryError {
    /// No font response bodies were supplied.
    #[error("CSS font-face registry has no embedded resources")]
    NoResources,
    /// No CSS face rules were supplied.
    #[error("CSS font-face registry has no face rules")]
    NoFaces,
    /// A resource identity is not a safe deterministic alias component.
    #[error("invalid embedded font resource id {0:?}")]
    InvalidResourceId(String),
    /// Two embedded bodies share an identity.
    #[error("duplicate embedded font resource id {0}")]
    DuplicateResource(String),
    /// An embedded body contains no bytes.
    #[error("embedded font resource {0} is empty")]
    EmptyResource(String),
    /// A CSS face uses an empty family name.
    #[error("CSS font face has an empty family")]
    EmptyFamily,
    /// A CSS face weight descriptor is outside the CSS range.
    #[error("invalid CSS font weight range {start}..={end}")]
    InvalidWeightRange {
        /// Inclusive lower CSS weight bound.
        start: u16,
        /// Inclusive upper CSS weight bound.
        end: u16,
    },
    /// A CSS face contains no coverage descriptor.
    #[error("CSS font face {family} has no Unicode ranges")]
    NoUnicodeRanges {
        /// Public CSS family name of the invalid face.
        family: String,
    },
    /// A CSS range is reversed or outside Unicode.
    #[error("invalid CSS Unicode range U+{start:04X}-U+{end:04X}")]
    InvalidUnicodeRange {
        /// Inclusive lower Unicode code point.
        start: u32,
        /// Inclusive upper Unicode code point.
        end: u32,
    },
    /// A CSS face references a body that was not supplied.
    #[error("CSS font face references unknown resource {0}")]
    UnknownResource(String),
    /// A supplied body has no face declaration.
    #[error("embedded font resource {0} is unused")]
    UnusedResource(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(id: &'static str) -> EmbeddedFontResource {
        EmbeddedFontResource::new(id, Cow::Borrowed(&[1, 2, 3]))
    }

    fn face(resource_id: &'static str) -> CssFontFace {
        CssFontFace::new(
            "Waypath Test",
            FontStyle::Normal,
            400..=700,
            Arc::from([CssUnicodeRange::new(0x20, 0x7e).unwrap()]),
            resource_id,
        )
    }

    #[test]
    fn registry_validates_exact_resources_faces_and_ranges() {
        let registry =
            CssFontFaceRegistry::new(vec![resource("latin")], vec![face("latin")]).unwrap();
        assert_eq!(registry.resources().len(), 1);
        assert_eq!(registry.faces().len(), 1);
        assert!(registry.faces()[0].covers('A'));
        assert!(!registry.faces()[0].covers('λ'));
        assert_eq!(
            registry.resources()[0].native_family_alias(),
            ".GPUIEmbeddedFont.latin"
        );
    }

    #[test]
    fn registry_rejects_unknown_duplicate_unused_and_invalid_inputs() {
        assert_eq!(
            CssFontFaceRegistry::new(vec![resource("latin")], vec![face("missing")]).unwrap_err(),
            CssFontFaceRegistryError::UnknownResource("missing".into())
        );
        assert_eq!(
            CssFontFaceRegistry::new(
                vec![resource("latin"), resource("latin")],
                vec![face("latin")]
            )
            .unwrap_err(),
            CssFontFaceRegistryError::DuplicateResource("latin".into())
        );
        assert_eq!(
            CssFontFaceRegistry::new(
                vec![resource("latin"), resource("unused")],
                vec![face("latin")]
            )
            .unwrap_err(),
            CssFontFaceRegistryError::UnusedResource("unused".into())
        );
        assert_eq!(
            CssUnicodeRange::new(0x110000, 0x110000).unwrap_err(),
            CssFontFaceRegistryError::InvalidUnicodeRange {
                start: 0x110000,
                end: 0x110000,
            }
        );
    }
}

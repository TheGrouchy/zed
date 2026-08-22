use crate::{DirectWriteTextSystem, DirectXDevices};
use gpui::{
    CssFontFace, CssFontFaceRegistry, CssUnicodeRange, EmbeddedFontResource, FontDisplayEventKind,
    FontDisplayFace, FontDisplayStatus, FontDisplaySwap, FontStyle, FontWeight, Pixels, TextRun,
    TextSystem, WindowTextSystem, font,
};
use std::{borrow::Cow, sync::Arc};

const LILEX: &[u8] = include_bytes!("../../../assets/fonts/lilex/Lilex-Regular.ttf");
const SOURCE_FACES: [(&str, &str, u16); 5] = [
    (
        "font_import_889af3b5eece50dc4c030f08",
        "Big Shoulders Display",
        700,
    ),
    (
        "font_import_8d4e5813834ad836509f6f4b",
        "JetBrains Mono",
        500,
    ),
    ("font_import_d3b3c4d08167798caa87510f", "Space Grotesk", 600),
    ("font_ac982175fcb49d977cf49a7e", "Sudo", 700),
    ("font_4249aa1efc05f71021974de3", "Sudo Outlined", 700),
];

fn unloaded_font_system() -> (Arc<TextSystem>, WindowTextSystem) {
    let devices = DirectXDevices::new().unwrap();
    let platform = Arc::new(DirectWriteTextSystem::new(&devices).unwrap());
    let text_system = Arc::new(TextSystem::new(platform));
    let window_text_system = WindowTextSystem::new(text_system.clone());
    (text_system, window_text_system)
}

fn source_run(text: &str, family: &str, weight: u16) -> TextRun {
    let mut source_font = font(family);
    source_font.weight = FontWeight(weight.into());
    TextRun {
        len: text.len(),
        font: source_font,
        ..TextRun::default()
    }
}

fn css_registry(
    resource_id: &str,
    family: &str,
    weight: u16,
    bytes: Cow<'static, [u8]>,
) -> CssFontFaceRegistry {
    CssFontFaceRegistry::new(
        vec![EmbeddedFontResource::new(resource_id, bytes)],
        vec![CssFontFace::new(
            family,
            FontStyle::Normal,
            weight..=weight,
            Arc::from([CssUnicodeRange::new(0x20, 0x7e).unwrap()]),
            resource_id,
        )],
    )
    .unwrap()
}

#[test]
fn delayed_registration_reflows_already_shaped_text_for_all_source_face_ids() {
    const TEXT: &str = "WAYPATH 0123456789";

    for (source_face_id, family, weight) in SOURCE_FACES {
        let (text_system, window_text_system) = unloaded_font_system();
        let run = source_run(TEXT, family, weight);
        let before_resolved_id = text_system.resolve_font(&run.font);
        let before = window_text_system.layout_line(
            TEXT,
            Pixels::from(32.0),
            std::slice::from_ref(&run),
            None,
        );
        assert_ne!(
            text_system
                .get_font_for_id(before_resolved_id)
                .unwrap()
                .family
                .as_str(),
            family
        );

        let lifecycle = FontDisplaySwap::new(text_system.clone());
        let (token, _) = lifecycle
            .begin(FontDisplayFace::new(source_face_id, family).unwrap())
            .unwrap();
        lifecycle
            .complete_registration(token, |text_system| {
                text_system.add_css_font_faces(css_registry(
                    source_face_id,
                    family,
                    weight,
                    Cow::Borrowed(LILEX),
                ))
            })
            .unwrap();

        let after_resolved_id = text_system.resolve_font(&run.font);
        let after = window_text_system.layout_line(TEXT, Pixels::from(32.0), &[run], None);
        assert_ne!(before_resolved_id, after_resolved_id);
        assert_eq!(
            text_system
                .get_font_for_id(after_resolved_id)
                .unwrap()
                .family
                .as_str(),
            family
        );
        assert_ne!(before.width, after.width);
        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(lifecycle.status(source_face_id), FontDisplayStatus::Loaded);
    }
}

#[test]
fn invalid_font_bytes_emit_error_and_keep_fallback_shaping_stable() {
    const TEXT: &str = "WAYPATH 0123456789";
    let (source_face_id, family, weight) = SOURCE_FACES[0];
    let (text_system, window_text_system) = unloaded_font_system();
    let run = source_run(TEXT, family, weight);
    let before =
        window_text_system.layout_line(TEXT, Pixels::from(32.0), std::slice::from_ref(&run), None);
    let lifecycle = FontDisplaySwap::new(text_system);
    let (token, _) = lifecycle
        .begin(FontDisplayFace::new(source_face_id, family).unwrap())
        .unwrap();

    let event = lifecycle
        .complete_registration(token, |text_system| {
            text_system.add_css_font_faces(css_registry(
                source_face_id,
                family,
                weight,
                Cow::Borrowed(b"not a font"),
            ))
        })
        .unwrap();

    assert!(matches!(event.kind, FontDisplayEventKind::Error { .. }));
    let after = window_text_system.layout_line(TEXT, Pixels::from(32.0), &[run], None);
    assert_eq!(after.runs[0].font_id, before.runs[0].font_id);
    assert_eq!(after.width, before.width);
    assert!(Arc::ptr_eq(&before, &after));
}

#[test]
fn failed_resource_preserves_fallback_layout_and_negative_cache() {
    const TEXT: &str = "WAYPATH 0123456789";
    let (source_face_id, family, weight) = SOURCE_FACES[0];
    let (text_system, window_text_system) = unloaded_font_system();
    let run = source_run(TEXT, family, weight);
    let before =
        window_text_system.layout_line(TEXT, Pixels::from(32.0), std::slice::from_ref(&run), None);
    let before_font_id = before.runs[0].font_id;
    let lifecycle = FontDisplaySwap::new(text_system);
    let (token, _) = lifecycle
        .begin(FontDisplayFace::new(source_face_id, family).unwrap())
        .unwrap();

    lifecycle.fail(token, "locked 404").unwrap();

    let after = window_text_system.layout_line(TEXT, Pixels::from(32.0), &[run], None);
    assert_eq!(after.runs[0].font_id, before_font_id);
    assert_eq!(after.width, before.width);
    assert!(Arc::ptr_eq(&before, &after));
    assert!(matches!(
        lifecycle.status(source_face_id),
        FontDisplayStatus::Error { .. }
    ));
}

use crate::{SharedString, TextRun, WhiteSpace};
use thiserror::Error;

/// The CSS default tab interval, expressed in preserved space characters.
pub const CSS_TAB_SIZE: usize = 8;

/// Text and style runs after deterministic CSS whitespace processing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedWhitespace {
    /// The normalized and collapsed text passed to every platform shaper.
    pub text: SharedString,
    /// Style runs remapped to cover the processed UTF-8 bytes exactly.
    pub runs: Vec<TextRun>,
}

/// A fail-closed error produced before platform shaping.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WhitespaceError {
    /// Style runs do not cover the source text exactly.
    #[error("text runs cover {runs_len} bytes but source contains {text_len} bytes")]
    RunCoverage {
        /// Source UTF-8 byte length.
        text_len: usize,
        /// Total UTF-8 byte coverage from the supplied runs.
        runs_len: usize,
    },
    /// A style boundary splits a UTF-8 character.
    #[error("text run ends at non-character boundary {index}")]
    RunBoundary {
        /// Invalid UTF-8 byte boundary.
        index: usize,
    },
}

struct OutputBuilder<'a> {
    text: String,
    runs: Vec<TextRun>,
    source_runs: &'a [TextRun],
}

impl<'a> OutputBuilder<'a> {
    fn new(source_runs: &'a [TextRun]) -> Self {
        Self {
            text: String::new(),
            runs: Vec::new(),
            source_runs,
        }
    }

    fn push(&mut self, value: &str, source_run: usize) {
        if value.is_empty() {
            return;
        }
        let source = &self.source_runs[source_run];
        if let Some(last) = self.runs.last_mut()
            && same_style(last, source)
        {
            last.len += value.len();
        } else {
            let mut run = source.clone();
            run.len = value.len();
            self.runs.push(run);
        }
        self.text.push_str(value);
    }

    fn finish(self) -> PreparedWhitespace {
        PreparedWhitespace {
            text: self.text.into(),
            runs: self.runs,
        }
    }
}

fn same_style(left: &TextRun, right: &TextRun) -> bool {
    left.font == right.font
        && left.color == right.color
        && left.background_color == right.background_color
        && left.underline == right.underline
        && left.strikethrough == right.strikethrough
        && left.letter_spacing == right.letter_spacing
}

fn validate_runs(text: &str, runs: &[TextRun]) -> Result<(), WhitespaceError> {
    let runs_len = runs.iter().map(|run| run.len).sum::<usize>();
    if runs_len != text.len() {
        return Err(WhitespaceError::RunCoverage {
            text_len: text.len(),
            runs_len,
        });
    }
    let mut end = 0;
    for run in runs {
        end += run.len;
        if !text.is_char_boundary(end) {
            return Err(WhitespaceError::RunBoundary { index: end });
        }
    }
    Ok(())
}

fn run_for_offset(
    runs: &[TextRun],
    offset: usize,
    run_index: &mut usize,
    run_end: &mut usize,
) -> usize {
    while offset >= *run_end && *run_index + 1 < runs.len() {
        *run_index += 1;
        *run_end += runs[*run_index].len;
    }
    *run_index
}

/// Normalize source segment breaks and apply one of the exact reachable CSS
/// whitespace modes before text is handed to DirectWrite, Cosmic Text, or
/// CoreText.
///
/// CSS collapsible characters are TAB, LF, and SPACE. Other Unicode
/// separators, including NBSP, are preserved. Chromium treats CRLF as one LF
/// segment break in DOM text, while isolated CR and FF are zero-width controls;
/// those controls are therefore removed rather than fabricated into lines.
/// Tabs in `pre-wrap` advance to the next eight-column stop; in collapsing
/// modes they participate in the collapsible run.
pub fn prepare_whitespace(
    text: SharedString,
    runs: Vec<TextRun>,
    mode: WhiteSpace,
) -> Result<PreparedWhitespace, WhitespaceError> {
    validate_runs(&text, &runs)?;
    if !mode.uses_css_processing() || text.is_empty() {
        return Ok(PreparedWhitespace { text, runs });
    }

    let bytes = text.as_bytes();
    let mut output = OutputBuilder::new(&runs);
    let mut offset = 0;
    let mut run_index = 0;
    let mut run_end = runs.first().map_or(0, |run| run.len);
    let mut pending_space_run = None;

    while offset < bytes.len() {
        let source_run = run_for_offset(&runs, offset, &mut run_index, &mut run_end);
        let remaining = &text[offset..];
        let character = remaining.chars().next().expect("offset is in bounds");
        let mut consumed = character.len_utf8();
        let (segment_break, zero_width_control) = match character {
            '\r' => {
                if remaining.as_bytes().get(1) == Some(&b'\n') {
                    consumed += 1;
                    (true, false)
                } else {
                    (false, true)
                }
            }
            '\n' => (true, false),
            '\u{000c}' => (false, true),
            _ => (false, false),
        };

        if zero_width_control {
            offset += consumed;
            continue;
        }

        match mode {
            WhiteSpace::Normal | WhiteSpace::Nowrap => {
                if segment_break || matches!(character, ' ' | '\t') {
                    if !output.text.is_empty() && pending_space_run.is_none() {
                        pending_space_run = Some(source_run);
                    }
                } else {
                    if let Some(space_run) = pending_space_run.take() {
                        output.push(" ", space_run);
                    }
                    output.push(&remaining[..consumed], source_run);
                }
            }
            WhiteSpace::PreLine => {
                if segment_break {
                    pending_space_run = None;
                    output.push("\n", source_run);
                } else if matches!(character, ' ' | '\t') {
                    if !output.text.ends_with('\n')
                        && !output.text.is_empty()
                        && pending_space_run.is_none()
                    {
                        pending_space_run = Some(source_run);
                    }
                } else {
                    if let Some(space_run) = pending_space_run.take() {
                        output.push(" ", space_run);
                    }
                    output.push(&remaining[..consumed], source_run);
                }
            }
            WhiteSpace::PreWrap => {
                if segment_break {
                    output.push("\n", source_run);
                } else if character == '\t' {
                    // Keep the tab as a semantic character. The shared line
                    // layout pass advances it to the next pixel stop using
                    // eight shaped U+0020 advances in the tab's own font run.
                    output.push("\t", source_run);
                } else {
                    output.push(&remaining[..consumed], source_run);
                }
            }
            WhiteSpace::Legacy => unreachable!("legacy returns before processing"),
        }
        offset += consumed;
    }

    Ok(output.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TextStyle, blue, font, red};
    use serde_json::Value;

    fn run(len: usize) -> TextRun {
        TextRun {
            len,
            font: font(".ZedMono"),
            ..Default::default()
        }
    }

    #[test]
    fn css_wrap_policy_is_explicit_and_legacy_remains_the_default() {
        assert_eq!(TextStyle::default().white_space, WhiteSpace::Legacy);
        assert!(WhiteSpace::Legacy.permits_soft_wrap());
        assert!(WhiteSpace::Normal.permits_soft_wrap());
        assert!(!WhiteSpace::Nowrap.permits_soft_wrap());
        assert!(WhiteSpace::PreLine.permits_soft_wrap());
        assert!(WhiteSpace::PreWrap.permits_soft_wrap());
        assert!(!WhiteSpace::Legacy.uses_css_processing());
        assert!(WhiteSpace::Normal.uses_css_processing());
    }

    #[test]
    fn four_reachable_modes_process_every_css_segment_break() {
        let source: SharedString = "\t alpha \r\n  beta\u{000c}\tgamma  ".into();
        let source_run = vec![run(source.len())];

        for mode in [WhiteSpace::Normal, WhiteSpace::Nowrap] {
            let prepared = prepare_whitespace(source.clone(), source_run.clone(), mode).unwrap();
            assert_eq!(prepared.text, "alpha beta gamma");
        }
        let pre_line =
            prepare_whitespace(source.clone(), source_run.clone(), WhiteSpace::PreLine).unwrap();
        assert_eq!(pre_line.text, "alpha\nbeta gamma");
        let pre_wrap = prepare_whitespace(source, source_run, WhiteSpace::PreWrap).unwrap();
        assert_eq!(pre_wrap.text, "\t alpha \n  beta\tgamma  ");
    }

    #[test]
    fn pre_wrap_tabs_advance_to_eight_column_stops() {
        let source: SharedString = "ab\tc\t\td".into();
        let prepared =
            prepare_whitespace(source.clone(), vec![run(source.len())], WhiteSpace::PreWrap)
                .unwrap();
        assert_eq!(prepared.text, source);
    }

    #[test]
    fn collapsed_and_expanded_bytes_remap_style_runs_exactly() {
        let source: SharedString = "a  \tb".into();
        let mut first = run(1);
        first.color = red();
        let mut whitespace = run(3);
        whitespace.color = blue();
        let mut last = run(1);
        last.color = red();
        let prepared =
            prepare_whitespace(source, vec![first, whitespace, last], WhiteSpace::Normal).unwrap();
        assert_eq!(prepared.text, "a b");
        assert_eq!(
            prepared.runs.iter().map(|run| run.len).collect::<Vec<_>>(),
            [1, 1, 1]
        );
        assert_eq!(
            prepared
                .runs
                .iter()
                .map(|run| run.color)
                .collect::<Vec<_>>(),
            [red(), blue(), red()]
        );
    }

    #[test]
    fn invalid_run_coverage_and_utf8_boundaries_fail_closed() {
        let text: SharedString = "é".into();
        assert_eq!(
            prepare_whitespace(text.clone(), vec![run(1)], WhiteSpace::Normal),
            Err(WhitespaceError::RunCoverage {
                text_len: 2,
                runs_len: 1,
            })
        );
        assert_eq!(
            prepare_whitespace(text, vec![run(1), run(1)], WhiteSpace::Normal),
            Err(WhitespaceError::RunBoundary { index: 1 })
        );
    }

    #[test]
    fn locked_declarations_and_adversarial_strings_match_chromium() {
        let oracle: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/css_whitespace_chromium.json"
        ))
        .unwrap();
        assert_eq!(oracle["devicePixelRatio"], 1.25);
        assert_eq!(oracle["forcedDeviceScaleFactor"], 1.25);
        assert_eq!(
            oracle["declarationCounts"],
            serde_json::json!({
                "normal": 2,
                "nowrap": 21,
                "pre-line": 1,
                "pre-wrap": 2,
            })
        );
        assert_eq!(oracle["declarations"].as_array().unwrap().len(), 26);
        assert_eq!(oracle["declarationCases"].as_array().unwrap().len(), 26);
        for case in oracle["declarationCases"].as_array().unwrap() {
            assert_eq!(case["mode"], case["computedMode"]);
        }

        for case in oracle["adversarialCases"].as_array().unwrap() {
            let mode = match case["mode"].as_str().unwrap() {
                "normal" => WhiteSpace::Normal,
                "nowrap" => WhiteSpace::Nowrap,
                "pre-line" => WhiteSpace::PreLine,
                "pre-wrap" => WhiteSpace::PreWrap,
                other => panic!("unexpected Chromium mode {other}"),
            };
            let source: SharedString = case["source"].as_str().unwrap().into();
            let prepared =
                prepare_whitespace(source.clone(), vec![run(source.len())], mode).unwrap();
            let chromium = case["innerText"]
                .as_str()
                .unwrap()
                .replace("\r\n", "\n")
                .replace(['\r', '\u{000c}'], "");
            assert_eq!(prepared.text, chromium, "{} {mode:?}", case["name"]);
        }

        let tabs = oracle["adversarialCases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["mode"] == "pre-wrap" && case["name"] == "tabs")
            .unwrap();
        let characters = tabs["characters"].as_array().unwrap();
        let glyph_width = characters[0][0]["width"].as_f64().unwrap();
        let first_tab = characters[1][0]["width"].as_f64().unwrap();
        let second_tab = characters[3][0]["width"].as_f64().unwrap();
        let third_tab = characters[4][0]["width"].as_f64().unwrap();
        assert!((first_tab / glyph_width - 7.0).abs() < 0.01);
        assert!((second_tab / glyph_width - 7.0).abs() < 0.01);
        assert!((third_tab / glyph_width - 8.0).abs() < 0.01);
        assert_eq!(tabs["lineTops"].as_array().unwrap().len(), 2);
    }
}

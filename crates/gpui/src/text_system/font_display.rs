use super::TextSystem;
use crate::{App, Result, SharedString};
use anyhow::{anyhow, bail};
use async_channel::{Receiver, Sender};
use collections::FxHashMap;
use parking_lot::Mutex;
use std::{borrow::Cow, fmt, future::Future, sync::Arc, time::Duration};
use web_time::Instant;

/// Supplies monotonic timestamps for font-display lifecycle events.
///
/// Applications can inject a frozen clock to make resource completion order
/// reproducible in tests.
pub trait FontDisplayClock: Send + Sync + 'static {
    /// Returns the elapsed time used to timestamp the next lifecycle event.
    fn now(&self) -> Duration;
}

struct MonotonicFontDisplayClock {
    origin: Instant,
}

impl Default for MonotonicFontDisplayClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl FontDisplayClock for MonotonicFontDisplayClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

/// Stable identity and family metadata for one delayed font resource.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FontDisplayFace {
    /// Stable resource identity supplied by the caller.
    pub id: SharedString,
    /// Font family whose failed resolutions may be retried after registration.
    pub family: SharedString,
}

impl FontDisplayFace {
    /// Constructs a face descriptor, rejecting empty identities and families.
    pub fn new(id: impl Into<SharedString>, family: impl Into<SharedString>) -> Result<Self> {
        let face = Self {
            id: id.into(),
            family: family.into(),
        };
        if face.id.is_empty() {
            bail!("font-display face id cannot be empty");
        }
        if face.family.is_empty() {
            bail!("font-display face family cannot be empty");
        }
        Ok(face)
    }
}

/// Observable state of a delayed font resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FontDisplayStatus {
    /// No request has been started for the face.
    Unrequested,
    /// The resource is pending and fallback text remains visible.
    Loading,
    /// The resource was registered and the primary face is eligible for shaping.
    Loaded,
    /// The resource or platform registration failed and fallback text remains visible.
    Error {
        /// Diagnostic text supplied by the resource or platform loader.
        message: SharedString,
    },
}

impl FontDisplayStatus {
    /// Returns whether source-equivalent swap semantics require primary or fallback text.
    pub fn visible_face(&self) -> FontDisplayVisibleFace {
        match self {
            Self::Loaded => FontDisplayVisibleFace::Primary,
            Self::Unrequested | Self::Loading | Self::Error { .. } => {
                FontDisplayVisibleFace::Fallback
            }
        }
    }
}

/// Which face must remain visible for a lifecycle status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FontDisplayVisibleFace {
    /// The declared primary font is visible.
    Primary,
    /// The next available fallback font is visible.
    Fallback,
}

/// Kind of a font-display lifecycle event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FontDisplayEventKind {
    /// A delayed resource request started.
    Loading,
    /// The resource was registered successfully.
    Loaded,
    /// Loading or platform registration failed.
    Error {
        /// Diagnostic text supplied by the resource or platform loader.
        message: SharedString,
    },
}

/// Deterministically ordered event emitted by [`FontDisplaySwap`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FontDisplayEvent {
    /// Monotonic sequence number assigned while holding the lifecycle lock.
    pub sequence: u64,
    /// Timestamp from the injected lifecycle clock.
    pub timestamp: Duration,
    /// Face affected by the transition.
    pub face: FontDisplayFace,
    /// Transition that occurred.
    pub kind: FontDisplayEventKind,
}

/// Single-use authority to complete a particular loading generation.
#[derive(Debug)]
pub struct FontDisplayLoadToken {
    face: FontDisplayFace,
    generation: u64,
}

#[derive(Clone, Debug)]
struct FaceState {
    face: FontDisplayFace,
    generation: u64,
    status: FontDisplayStatus,
}

struct RegistryState {
    faces: FxHashMap<SharedString, FaceState>,
    subscribers: Vec<Sender<FontDisplayEvent>>,
    next_sequence: u64,
}

impl Default for RegistryState {
    fn default() -> Self {
        Self {
            faces: FxHashMap::default(),
            subscribers: Vec::new(),
            next_sequence: 1,
        }
    }
}

struct FontDisplaySwapInner {
    text_system: Arc<TextSystem>,
    clock: Arc<dyn FontDisplayClock>,
    registry: Mutex<RegistryState>,
}

/// Coordinates delayed font registration with swap lifecycle state.
#[derive(Clone)]
pub struct FontDisplaySwap {
    inner: Arc<FontDisplaySwapInner>,
}

impl fmt::Debug for FontDisplaySwap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FontDisplaySwap")
            .finish_non_exhaustive()
    }
}

impl FontDisplaySwap {
    /// Creates a lifecycle bound to a text system and a production monotonic clock.
    pub fn new(text_system: Arc<TextSystem>) -> Self {
        Self::with_clock(text_system, Arc::new(MonotonicFontDisplayClock::default()))
    }

    /// Creates a lifecycle using an injected deterministic clock.
    pub fn with_clock(text_system: Arc<TextSystem>, clock: Arc<dyn FontDisplayClock>) -> Self {
        Self {
            inner: Arc::new(FontDisplaySwapInner {
                text_system,
                clock,
                registry: Mutex::new(RegistryState::default()),
            }),
        }
    }

    /// Subscribes to future loading, loaded, and error transitions.
    pub fn subscribe(&self) -> Receiver<FontDisplayEvent> {
        let (sender, receiver) = async_channel::unbounded();
        self.inner.registry.lock().subscribers.push(sender);
        receiver
    }

    /// Returns the current status for a resource identity.
    pub fn status(&self, face_id: &str) -> FontDisplayStatus {
        self.inner
            .registry
            .lock()
            .faces
            .get(face_id)
            .map(|state| state.status.clone())
            .unwrap_or(FontDisplayStatus::Unrequested)
    }

    /// Starts one loading generation and emits its loading event.
    pub fn begin(&self, face: FontDisplayFace) -> Result<(FontDisplayLoadToken, FontDisplayEvent)> {
        let mut registry = self.inner.registry.lock();
        let generation = match registry.faces.get(&face.id) {
            Some(state) if state.face.family != face.family => bail!(
                "font-display face '{}' was already registered for family '{}'",
                face.id,
                state.face.family
            ),
            Some(state) if state.status == FontDisplayStatus::Loading => {
                bail!("font-display face '{}' is already loading", face.id)
            }
            Some(state) if state.status == FontDisplayStatus::Loaded => {
                bail!("font-display face '{}' is already loaded", face.id)
            }
            Some(state) => state
                .generation
                .checked_add(1)
                .ok_or_else(|| anyhow!("font-display generation overflow for '{}'", face.id))?,
            None => 1,
        };

        registry.faces.insert(
            face.id.clone(),
            FaceState {
                face: face.clone(),
                generation,
                status: FontDisplayStatus::Loading,
            },
        );
        let event = self.emit_locked(&mut registry, face.clone(), FontDisplayEventKind::Loading)?;
        Ok((FontDisplayLoadToken { face, generation }, event))
    }

    /// Registers loaded font bytes, retries failed family resolutions, and emits loaded.
    pub fn complete(
        &self,
        token: FontDisplayLoadToken,
        fonts: Vec<Cow<'static, [u8]>>,
    ) -> Result<FontDisplayEvent> {
        if fonts.is_empty() {
            return self.fail(token, "font-display resource produced no font bytes");
        }
        self.complete_registration(token, move |text_system| text_system.add_fonts(fonts))
    }

    /// Runs a caller-supplied face registrar before cache invalidation and loaded emission.
    ///
    /// This keeps lifecycle ordering independent from the platform's face registry format.
    /// A CSS face registrar can atomically bind descriptors and bytes here without making
    /// the lifecycle depend on CSS matching or platform font-selection internals.
    pub fn complete_registration(
        &self,
        token: FontDisplayLoadToken,
        register: impl FnOnce(&TextSystem) -> Result<()>,
    ) -> Result<FontDisplayEvent> {
        self.validate_token(&token)?;
        if let Err(error) = register(&self.inner.text_system) {
            return self.fail(token, format!("font registration failed: {error}"));
        }

        self.inner
            .text_system
            .clear_failed_font_resolutions(&token.face.family);
        self.finish(
            token,
            FontDisplayStatus::Loaded,
            FontDisplayEventKind::Loaded,
        )
    }

    /// Records a resource failure without exposing the unavailable face to shaping.
    pub fn fail(
        &self,
        token: FontDisplayLoadToken,
        error: impl fmt::Display,
    ) -> Result<FontDisplayEvent> {
        let message = SharedString::new(error.to_string());
        self.finish(
            token,
            FontDisplayStatus::Error {
                message: message.clone(),
            },
            FontDisplayEventKind::Error { message },
        )
    }

    /// Starts and detaches a delayed resource request on the GPUI foreground executor.
    ///
    /// The terminal transition always refreshes application windows so text shaped with
    /// a fallback is resolved and measured again after success or stable failure.
    pub fn load<F>(
        &self,
        face: FontDisplayFace,
        resource: F,
        cx: &mut App,
    ) -> Result<FontDisplayEvent>
    where
        F: Future<Output = Result<Vec<Cow<'static, [u8]>>>> + 'static,
    {
        self.load_with_registration(
            face,
            resource,
            |text_system, fonts| text_system.add_fonts(fonts),
            cx,
        )
    }

    /// Starts a delayed resource request with a caller-supplied platform registrar.
    ///
    /// Registration, failed-resolution eviction, loaded emission, and window refresh
    /// remain ordered even when another GPUI module owns the face descriptor registry.
    pub fn load_with_registration<F, R>(
        &self,
        face: FontDisplayFace,
        resource: F,
        register: R,
        cx: &mut App,
    ) -> Result<FontDisplayEvent>
    where
        F: Future<Output = Result<Vec<Cow<'static, [u8]>>>> + 'static,
        R: FnOnce(&TextSystem, Vec<Cow<'static, [u8]>>) -> Result<()> + 'static,
    {
        let (token, loading) = self.begin(face)?;
        let lifecycle = self.clone();
        cx.spawn(async move |cx| {
            let terminal = match resource.await {
                Ok(fonts) if fonts.is_empty() => {
                    lifecycle.fail(token, "font-display resource produced no font bytes")
                }
                Ok(fonts) => lifecycle
                    .complete_registration(token, move |text_system| register(text_system, fonts)),
                Err(error) => lifecycle.fail(token, error),
            };
            if let Err(error) = terminal {
                log::error!("font-display completion was rejected: {error:#}");
            }
            cx.refresh();
        })
        .detach();
        cx.refresh_windows();
        Ok(loading)
    }

    fn validate_token(&self, token: &FontDisplayLoadToken) -> Result<()> {
        let registry = self.inner.registry.lock();
        let state = registry
            .faces
            .get(&token.face.id)
            .ok_or_else(|| anyhow!("unknown font-display face '{}'", token.face.id))?;
        if state.face.family != token.face.family
            || state.generation != token.generation
            || state.status != FontDisplayStatus::Loading
        {
            bail!(
                "stale font-display completion for '{}' generation {}",
                token.face.id,
                token.generation
            );
        }
        Ok(())
    }

    fn finish(
        &self,
        token: FontDisplayLoadToken,
        status: FontDisplayStatus,
        kind: FontDisplayEventKind,
    ) -> Result<FontDisplayEvent> {
        let mut registry = self.inner.registry.lock();
        let state = registry
            .faces
            .get_mut(&token.face.id)
            .ok_or_else(|| anyhow!("unknown font-display face '{}'", token.face.id))?;
        if state.face.family != token.face.family
            || state.generation != token.generation
            || state.status != FontDisplayStatus::Loading
        {
            bail!(
                "stale font-display completion for '{}' generation {}",
                token.face.id,
                token.generation
            );
        }
        state.status = status;
        self.emit_locked(&mut registry, token.face, kind)
    }

    fn emit_locked(
        &self,
        registry: &mut RegistryState,
        face: FontDisplayFace,
        kind: FontDisplayEventKind,
    ) -> Result<FontDisplayEvent> {
        let sequence = registry.next_sequence;
        registry.next_sequence = registry
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("font-display event sequence overflow"))?;
        let event = FontDisplayEvent {
            sequence,
            timestamp: self.inner.clock.now(),
            face,
            kind,
        };
        registry
            .subscribers
            .retain(|subscriber| subscriber.try_send(event.clone()).is_ok());
        Ok(event)
    }
}

impl TextSystem {
    fn clear_failed_font_resolutions(&self, family: &str) -> usize {
        let mut cache = self.font_ids_by_font.write();
        let previous_len = cache.len();
        cache.retain(|font, result| {
            result.is_ok() || !font.family.as_ref().eq_ignore_ascii_case(family)
        });
        previous_len - cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FontDisplayVisibleFace, FontId, NoopTextSystem, TestApp, font};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    const SOURCE_FACES: [(&str, &str); 5] = [
        (
            "font_import_889af3b5eece50dc4c030f08",
            "Big Shoulders Display",
        ),
        ("font_import_8d4e5813834ad836509f6f4b", "JetBrains Mono"),
        ("font_import_d3b3c4d08167798caa87510f", "Space Grotesk"),
        ("font_ac982175fcb49d977cf49a7e", "Sudo"),
        ("font_4249aa1efc05f71021974de3", "Sudo Outlined"),
    ];

    #[derive(Default)]
    struct FrozenClock(AtomicU64);

    impl FrozenClock {
        fn set_millis(&self, millis: u64) {
            self.0.store(millis, Ordering::SeqCst);
        }
    }

    impl FontDisplayClock for FrozenClock {
        fn now(&self) -> Duration {
            Duration::from_millis(self.0.load(Ordering::SeqCst))
        }
    }

    fn lifecycle(clock: Arc<FrozenClock>) -> FontDisplaySwap {
        FontDisplaySwap::with_clock(
            Arc::new(TextSystem::new(Arc::new(NoopTextSystem::new()))),
            clock,
        )
    }

    fn drain_events(receiver: &Receiver<FontDisplayEvent>) -> Vec<FontDisplayEvent> {
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        events
    }

    #[test]
    fn five_source_faces_match_twenty_frozen_swap_oracle_facts() {
        let success_clock = Arc::new(FrozenClock::default());
        success_clock.set_millis(41);
        let success = lifecycle(success_clock);
        let success_events = success.subscribe();

        let failure_clock = Arc::new(FrozenClock::default());
        failure_clock.set_millis(73);
        let failure = lifecycle(failure_clock);
        let failure_events = failure.subscribe();

        for (index, (id, family)) in SOURCE_FACES.into_iter().enumerate() {
            let face = FontDisplayFace::new(id, family).unwrap();
            let (token, loading) = success.begin(face.clone()).unwrap();
            assert_eq!(
                success.status(id).visible_face(),
                FontDisplayVisibleFace::Fallback
            );
            let loaded = success
                .complete(token, vec![Cow::Borrowed(b"font fixture")])
                .unwrap();
            assert_eq!(
                success.status(id).visible_face(),
                FontDisplayVisibleFace::Primary
            );
            assert_eq!(loading.sequence, (index * 2 + 1) as u64);
            assert_eq!(loaded.sequence, (index * 2 + 2) as u64);
            assert_eq!(loading.timestamp, Duration::from_millis(41));
            assert_eq!(loaded.timestamp, Duration::from_millis(41));

            let (token, loading) = failure.begin(face).unwrap();
            assert_eq!(
                failure.status(id).visible_face(),
                FontDisplayVisibleFace::Fallback
            );
            let failed = failure.fail(token, "locked 404").unwrap();
            assert_eq!(
                failure.status(id).visible_face(),
                FontDisplayVisibleFace::Fallback
            );
            assert_eq!(loading.sequence, (index * 2 + 1) as u64);
            assert_eq!(failed.sequence, (index * 2 + 2) as u64);
            assert_eq!(loading.timestamp, Duration::from_millis(73));
            assert_eq!(failed.timestamp, Duration::from_millis(73));
        }

        let success_kinds = drain_events(&success_events)
            .into_iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>();
        let failure_kinds = drain_events(&failure_events)
            .into_iter()
            .map(|event| event.kind)
            .collect::<Vec<_>>();
        assert_eq!(success_kinds.len(), 10);
        assert_eq!(failure_kinds.len(), 10);
        for pair in success_kinds.chunks_exact(2) {
            assert_eq!(
                pair,
                [FontDisplayEventKind::Loading, FontDisplayEventKind::Loaded]
            );
        }
        for pair in failure_kinds.chunks_exact(2) {
            assert_eq!(pair[0], FontDisplayEventKind::Loading);
            assert!(matches!(pair[1], FontDisplayEventKind::Error { .. }));
        }
    }

    #[test]
    fn successful_registration_evicts_only_failed_matching_family_entries() {
        let text_system = Arc::new(TextSystem::new(Arc::new(NoopTextSystem::new())));
        text_system
            .font_ids_by_font
            .write()
            .insert(font("Sudo"), Err(anyhow!("not found")));
        text_system
            .font_ids_by_font
            .write()
            .insert(font("sudo").bold(), Err(anyhow!("not found")));
        text_system
            .font_ids_by_font
            .write()
            .insert(font("Other"), Err(anyhow!("not found")));
        text_system
            .font_ids_by_font
            .write()
            .insert(font("Preserved"), Ok(FontId(22)));
        let lifecycle =
            FontDisplaySwap::with_clock(text_system.clone(), Arc::new(FrozenClock::default()));
        let (token, _) = lifecycle
            .begin(FontDisplayFace::new("sudo", "Sudo").unwrap())
            .unwrap();

        lifecycle
            .complete(token, vec![Cow::Borrowed(b"font fixture")])
            .unwrap();

        let cache = text_system.font_ids_by_font.read();
        assert!(!cache.contains_key(&font("Sudo")));
        assert!(!cache.contains_key(&font("sudo").bold()));
        assert!(cache.contains_key(&font("Other")));
        assert_eq!(
            cache.get(&font("Preserved")).unwrap().as_ref().unwrap(),
            &FontId(22)
        );
    }

    #[test]
    fn failure_preserves_negative_cache_and_allows_explicit_retry() {
        let text_system = Arc::new(TextSystem::new(Arc::new(NoopTextSystem::new())));
        text_system
            .font_ids_by_font
            .write()
            .insert(font("Sudo"), Err(anyhow!("not found")));
        let lifecycle =
            FontDisplaySwap::with_clock(text_system.clone(), Arc::new(FrozenClock::default()));
        let face = FontDisplayFace::new("sudo", "Sudo").unwrap();
        let (first, _) = lifecycle.begin(face.clone()).unwrap();
        lifecycle.fail(first, "locked 404").unwrap();
        assert!(
            text_system
                .font_ids_by_font
                .read()
                .contains_key(&font("Sudo"))
        );

        let (retry, event) = lifecycle.begin(face).unwrap();
        assert_eq!(event.sequence, 3);
        lifecycle
            .complete(retry, vec![Cow::Borrowed(b"font fixture")])
            .unwrap();
        assert_eq!(lifecycle.status("sudo"), FontDisplayStatus::Loaded);
    }

    #[test]
    fn duplicate_family_and_stale_completion_are_rejected_without_state_drift() {
        let lifecycle = lifecycle(Arc::new(FrozenClock::default()));
        let face = FontDisplayFace::new("face", "Family").unwrap();
        let (_token, _) = lifecycle.begin(face.clone()).unwrap();
        let error = lifecycle
            .begin(FontDisplayFace::new("face", "Other Family").unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("already registered"));

        let stale = FontDisplayLoadToken {
            face,
            generation: 9,
        };
        let error = lifecycle.fail(stale, "late completion").unwrap_err();
        assert!(error.to_string().contains("stale"));
        assert_eq!(lifecycle.status("face"), FontDisplayStatus::Loading);
    }

    #[test]
    fn app_load_path_detaches_resource_and_reaches_terminal_event() {
        let clock = Arc::new(FrozenClock::default());
        clock.set_millis(19);
        let mut app = TestApp::new();
        let lifecycle =
            app.read(|cx| FontDisplaySwap::with_clock(cx.text_system().clone(), clock.clone()));
        let events = lifecycle.subscribe();
        let registration_called = Arc::new(AtomicBool::new(false));

        let loading = app.update(|cx| {
            let registration_called = registration_called.clone();
            lifecycle.load_with_registration(
                FontDisplayFace::new("detached", "Delayed Family").unwrap(),
                async { Ok(vec![Cow::Borrowed(b"font fixture" as &'static [u8])]) },
                move |_text_system, fonts| {
                    assert_eq!(fonts.len(), 1);
                    registration_called.store(true, Ordering::SeqCst);
                    Ok(())
                },
                cx,
            )
        });

        assert_eq!(loading.unwrap().kind, FontDisplayEventKind::Loading);
        assert!(registration_called.load(Ordering::SeqCst));
        assert_eq!(lifecycle.status("detached"), FontDisplayStatus::Loaded);
        let events = drain_events(&events);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, FontDisplayEventKind::Loading);
        assert_eq!(events[1].kind, FontDisplayEventKind::Loaded);
        assert_eq!(events[0].timestamp, Duration::from_millis(19));
        assert_eq!(events[1].timestamp, Duration::from_millis(19));
    }
}

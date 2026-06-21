//! Undo/redo history for user-driven scene edits.
//!
//! `History` is a bounded ring buffer of [`SceneCommand`] values. Each command
//! captures enough state to apply itself forward and produce its inverse, so
//! undo and redo are symmetric operations.
//!
//! Only mutations driven by user UI actions should funnel through `History` —
//! simulation ticks, agent-server pushes, and tele-op are not undoable.

use glam::Vec3;

use crate::scene::{AcousticMaterial, Listener, Scene, SceneObject, SoundSource};

/// A reversible mutation to [`Scene`].
///
/// Each variant carries enough state to apply forward AND to produce its
/// inverse via [`SceneCommand::invert`]. `Insert*` and `Remove*` use explicit
/// indices so deleting and re-inserting middle elements preserves ordering.
#[derive(Clone)]
pub enum SceneCommand {
    InsertSource {
        idx: usize,
        src: SoundSource,
    },
    RemoveSource {
        idx: usize,
        snapshot: SoundSource,
    },
    MoveSource {
        idx: usize,
        from: Vec3,
        to: Vec3,
    },
    SetSourceFreq {
        idx: usize,
        from: f32,
        to: f32,
    },
    SetSourcePower {
        idx: usize,
        from: f32,
        to: f32,
    },
    SetSourceEnabled {
        idx: usize,
        from: bool,
        to: bool,
    },

    InsertListener {
        idx: usize,
        listener: Listener,
    },
    RemoveListener {
        idx: usize,
        snapshot: Listener,
    },
    MoveListener {
        idx: usize,
        from: Vec3,
        to: Vec3,
    },

    InsertObject {
        idx: usize,
        obj: SceneObject,
    },
    RemoveObject {
        idx: usize,
        snapshot: SceneObject,
    },
    SetObjectVisible {
        idx: usize,
        from: bool,
        to: bool,
    },
    SetObjectMaterial {
        idx: usize,
        from: AcousticMaterial,
        to: AcousticMaterial,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryError {
    IndexOutOfBounds,
}

impl SceneCommand {
    /// Mutate `scene` to reflect this command. Returns
    /// [`HistoryError::IndexOutOfBounds`] if a target index no longer exists
    /// (e.g. the user removed an object behind the history's back via a
    /// non-tracked path).
    pub fn apply(&self, scene: &mut Scene) -> Result<(), HistoryError> {
        use SceneCommand::*;
        match self {
            InsertSource { idx, src } => {
                if *idx > scene.sound_sources.len() {
                    return Err(HistoryError::IndexOutOfBounds);
                }
                scene.sound_sources.insert(*idx, src.clone());
            }
            RemoveSource { idx, .. } => {
                if *idx >= scene.sound_sources.len() {
                    return Err(HistoryError::IndexOutOfBounds);
                }
                scene.sound_sources.remove(*idx);
            }
            MoveSource { idx, to, .. } => {
                let s = scene
                    .sound_sources
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                s.position = *to;
            }
            SetSourceFreq { idx, to, .. } => {
                let s = scene
                    .sound_sources
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                s.frequency_hz = *to;
            }
            SetSourcePower { idx, to, .. } => {
                let s = scene
                    .sound_sources
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                s.power_db = *to;
            }
            SetSourceEnabled { idx, to, .. } => {
                let s = scene
                    .sound_sources
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                s.enabled = *to;
            }

            InsertListener { idx, listener } => {
                if *idx > scene.listeners.len() {
                    return Err(HistoryError::IndexOutOfBounds);
                }
                scene.listeners.insert(*idx, listener.clone());
            }
            RemoveListener { idx, .. } => {
                if *idx >= scene.listeners.len() {
                    return Err(HistoryError::IndexOutOfBounds);
                }
                scene.listeners.remove(*idx);
            }
            MoveListener { idx, to, .. } => {
                let l = scene
                    .listeners
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                l.position = *to;
            }

            InsertObject { idx, obj } => {
                if *idx > scene.meshes.len() {
                    return Err(HistoryError::IndexOutOfBounds);
                }
                scene.meshes.insert(*idx, obj.clone());
            }
            RemoveObject { idx, .. } => {
                if *idx >= scene.meshes.len() {
                    return Err(HistoryError::IndexOutOfBounds);
                }
                scene.meshes.remove(*idx);
            }
            SetObjectVisible { idx, to, .. } => {
                let o = scene
                    .meshes
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                o.visible = *to;
            }
            SetObjectMaterial { idx, to, .. } => {
                let o = scene
                    .meshes
                    .get_mut(*idx)
                    .ok_or(HistoryError::IndexOutOfBounds)?;
                o.material = to.clone();
            }
        }
        Ok(())
    }

    /// The reverse of this command. `apply(invert) == undo`.
    pub fn invert(&self) -> SceneCommand {
        use SceneCommand::*;
        match self {
            InsertSource { idx, src } => RemoveSource {
                idx: *idx,
                snapshot: src.clone(),
            },
            RemoveSource { idx, snapshot } => InsertSource {
                idx: *idx,
                src: snapshot.clone(),
            },
            MoveSource { idx, from, to } => MoveSource {
                idx: *idx,
                from: *to,
                to: *from,
            },
            SetSourceFreq { idx, from, to } => SetSourceFreq {
                idx: *idx,
                from: *to,
                to: *from,
            },
            SetSourcePower { idx, from, to } => SetSourcePower {
                idx: *idx,
                from: *to,
                to: *from,
            },
            SetSourceEnabled { idx, from, to } => SetSourceEnabled {
                idx: *idx,
                from: *to,
                to: *from,
            },

            InsertListener { idx, listener } => RemoveListener {
                idx: *idx,
                snapshot: listener.clone(),
            },
            RemoveListener { idx, snapshot } => InsertListener {
                idx: *idx,
                listener: snapshot.clone(),
            },
            MoveListener { idx, from, to } => MoveListener {
                idx: *idx,
                from: *to,
                to: *from,
            },

            InsertObject { idx, obj } => RemoveObject {
                idx: *idx,
                snapshot: obj.clone(),
            },
            RemoveObject { idx, snapshot } => InsertObject {
                idx: *idx,
                obj: snapshot.clone(),
            },
            SetObjectVisible { idx, from, to } => SetObjectVisible {
                idx: *idx,
                from: *to,
                to: *from,
            },
            SetObjectMaterial { idx, from, to } => SetObjectMaterial {
                idx: *idx,
                from: to.clone(),
                to: from.clone(),
            },
        }
    }

    /// Short human-readable label, e.g. "Move source" — for the status bar
    /// "Undo X" / "Redo X" hint.
    pub fn name(&self) -> &'static str {
        use SceneCommand::*;
        match self {
            InsertSource { .. } => "Add source",
            RemoveSource { .. } => "Delete source",
            MoveSource { .. } => "Move source",
            SetSourceFreq { .. } => "Set source frequency",
            SetSourcePower { .. } => "Set source power",
            SetSourceEnabled { .. } => "Toggle source",
            InsertListener { .. } => "Add listener",
            RemoveListener { .. } => "Delete listener",
            MoveListener { .. } => "Move listener",
            InsertObject { .. } => "Add object",
            RemoveObject { .. } => "Delete object",
            SetObjectVisible { .. } => "Toggle visibility",
            SetObjectMaterial { .. } => "Set material",
        }
    }
}

/// Bounded undo/redo ring buffer.
///
/// `past` holds commands already applied to the scene (top = most recent).
/// `future` holds commands undone — ready to redo. Any new push clears
/// `future`, matching standard editor semantics.
pub struct History {
    past: Vec<SceneCommand>,
    future: Vec<SceneCommand>,
    capacity: usize,
}

impl History {
    pub const DEFAULT_CAPACITY: usize = 100;

    pub fn new(capacity: usize) -> Self {
        Self {
            past: Vec::new(),
            future: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// Apply `cmd` to `scene` and record it. Clears the redo stack.
    ///
    /// If `apply` fails (stale index), the command is dropped and no state
    /// changes. Returns the error so the caller can surface it.
    pub fn push(&mut self, cmd: SceneCommand, scene: &mut Scene) -> Result<(), HistoryError> {
        if let Err(e) = cmd.apply(scene) {
            // The command targets an index that no longer exists (the scene was
            // mutated behind the history's back). The scene is unchanged; log so
            // the dropped edit is never silently swallowed, and surface the
            // error to the caller.
            log::warn!("scene edit dropped — stale target index: {e:?}");
            return Err(e);
        }
        self.past.push(cmd);
        if self.past.len() > self.capacity {
            self.past.remove(0);
        }
        self.future.clear();
        Ok(())
    }

    /// Undo the most recent command. Returns the command's name (for UI hint)
    /// or `None` if there's nothing to undo / the inverse failed.
    pub fn undo(&mut self, scene: &mut Scene) -> Option<&'static str> {
        let cmd = self.past.pop()?;
        if cmd.invert().apply(scene).is_err() {
            // Stale state — drop the command rather than corrupt history.
            return None;
        }
        let name = cmd.name();
        self.future.push(cmd);
        Some(name)
    }

    /// Redo the most recently undone command.
    pub fn redo(&mut self, scene: &mut Scene) -> Option<&'static str> {
        let cmd = self.future.pop()?;
        if cmd.apply(scene).is_err() {
            return None;
        }
        let name = cmd.name();
        self.past.push(cmd);
        Some(name)
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    /// Peek at the most recent applied command's name — for the status bar
    /// "Undo: <name>" affordance.
    pub fn last_action_name(&self) -> Option<&'static str> {
        self.past.last().map(|c| c.name())
    }
    pub fn next_redo_name(&self) -> Option<&'static str> {
        self.future.last().map(|c| c.name())
    }

    pub fn clear(&mut self) {
        self.past.clear();
        self.future.clear();
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn past_len(&self) -> usize {
        self.past.len()
    }
    pub fn future_len(&self) -> usize {
        self.future.len()
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new(Self::DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;

    fn src(x: f32) -> SoundSource {
        SoundSource {
            position: Vec3::new(x, 0.0, 0.0),
            frequency_hz: 1000.0,
            power_db: 80.0,
            enabled: true,
        }
    }
    fn listener(x: f32) -> Listener {
        Listener {
            position: Vec3::new(x, 0.0, 0.0),
            name: format!("L{}", x as i32),
            capture_radius: 0.3,
        }
    }

    #[test]
    fn history_starts_empty() {
        let h = History::default();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.past_len(), 0);
        assert_eq!(h.future_len(), 0);
    }

    #[test]
    fn history_insert_source_then_undo_redo() {
        let mut scene = Scene::default();
        let mut h = History::new(10);

        h.push(
            SceneCommand::InsertSource {
                idx: 0,
                src: src(1.0),
            },
            &mut scene,
        )
        .unwrap();
        assert_eq!(scene.sound_sources.len(), 1);
        assert!(h.can_undo());
        assert!(!h.can_redo());

        let name = h.undo(&mut scene).unwrap();
        assert_eq!(name, "Add source");
        assert_eq!(scene.sound_sources.len(), 0);
        assert!(h.can_redo());

        let name = h.redo(&mut scene).unwrap();
        assert_eq!(name, "Add source");
        assert_eq!(scene.sound_sources.len(), 1);
        assert!((scene.sound_sources[0].position.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn history_move_source_round_trip() {
        let mut scene = Scene::default();
        scene.sound_sources.push(src(0.0));
        let mut h = History::new(10);

        h.push(
            SceneCommand::MoveSource {
                idx: 0,
                from: Vec3::ZERO,
                to: Vec3::new(5.0, 0.0, 0.0),
            },
            &mut scene,
        )
        .unwrap();
        assert!((scene.sound_sources[0].position.x - 5.0).abs() < 1e-6);

        h.undo(&mut scene).unwrap();
        assert!(scene.sound_sources[0].position.x.abs() < 1e-6);

        h.redo(&mut scene).unwrap();
        assert!((scene.sound_sources[0].position.x - 5.0).abs() < 1e-6);
    }

    #[test]
    fn history_remove_source_preserves_index() {
        let mut scene = Scene::default();
        scene.sound_sources.push(src(1.0));
        scene.sound_sources.push(src(2.0));
        scene.sound_sources.push(src(3.0));

        let mut h = History::new(10);
        let snap = scene.sound_sources[1].clone();
        h.push(
            SceneCommand::RemoveSource {
                idx: 1,
                snapshot: snap,
            },
            &mut scene,
        )
        .unwrap();
        assert_eq!(scene.sound_sources.len(), 2);
        assert!((scene.sound_sources[0].position.x - 1.0).abs() < 1e-6);
        assert!((scene.sound_sources[1].position.x - 3.0).abs() < 1e-6);

        // Undo: middle element comes back at idx 1.
        h.undo(&mut scene).unwrap();
        assert_eq!(scene.sound_sources.len(), 3);
        assert!((scene.sound_sources[1].position.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn history_push_clears_future_stack() {
        let mut scene = Scene::default();
        let mut h = History::new(10);

        h.push(
            SceneCommand::InsertSource {
                idx: 0,
                src: src(1.0),
            },
            &mut scene,
        )
        .unwrap();
        h.undo(&mut scene).unwrap();
        assert!(h.can_redo());

        // New action wipes the redo stack.
        h.push(
            SceneCommand::InsertSource {
                idx: 0,
                src: src(9.0),
            },
            &mut scene,
        )
        .unwrap();
        assert!(!h.can_redo());
        assert_eq!(scene.sound_sources.len(), 1);
        assert!((scene.sound_sources[0].position.x - 9.0).abs() < 1e-6);
    }

    #[test]
    fn history_respects_capacity() {
        let mut scene = Scene::default();
        let mut h = History::new(3);

        for i in 0..5 {
            h.push(
                SceneCommand::InsertSource {
                    idx: i,
                    src: src(i as f32),
                },
                &mut scene,
            )
            .unwrap();
        }
        assert_eq!(h.past_len(), 3);
        assert_eq!(h.capacity(), 3);
        assert_eq!(scene.sound_sources.len(), 5);

        // Only the last 3 are undoable — first two stick around in the scene.
        h.undo(&mut scene).unwrap();
        h.undo(&mut scene).unwrap();
        h.undo(&mut scene).unwrap();
        assert!(!h.can_undo());
        assert_eq!(scene.sound_sources.len(), 2);
    }

    #[test]
    fn history_listener_round_trip() {
        let mut scene = Scene::default();
        let mut h = History::new(10);

        h.push(
            SceneCommand::InsertListener {
                idx: 0,
                listener: listener(0.0),
            },
            &mut scene,
        )
        .unwrap();
        assert_eq!(scene.listeners.len(), 1);

        h.push(
            SceneCommand::MoveListener {
                idx: 0,
                from: Vec3::ZERO,
                to: Vec3::new(2.0, 0.0, 0.0),
            },
            &mut scene,
        )
        .unwrap();
        assert!((scene.listeners[0].position.x - 2.0).abs() < 1e-6);

        h.undo(&mut scene).unwrap(); // undo move
        assert!(scene.listeners[0].position.x.abs() < 1e-6);
        h.undo(&mut scene).unwrap(); // undo insert
        assert_eq!(scene.listeners.len(), 0);
    }

    #[test]
    fn history_set_source_freq_round_trip() {
        let mut scene = Scene::default();
        scene.sound_sources.push(src(0.0));
        let mut h = History::new(10);

        h.push(
            SceneCommand::SetSourceFreq {
                idx: 0,
                from: 1000.0,
                to: 2000.0,
            },
            &mut scene,
        )
        .unwrap();
        assert!((scene.sound_sources[0].frequency_hz - 2000.0).abs() < 1e-6);
        h.undo(&mut scene).unwrap();
        assert!((scene.sound_sources[0].frequency_hz - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn history_stale_index_drops_gracefully() {
        let mut scene = Scene::default();
        let mut h = History::new(10);

        // Try to move a source that doesn't exist — should error, no panic.
        let result = h.push(
            SceneCommand::MoveSource {
                idx: 0,
                from: Vec3::ZERO,
                to: Vec3::ONE,
            },
            &mut scene,
        );
        assert_eq!(result, Err(HistoryError::IndexOutOfBounds));
        assert!(!h.can_undo());
    }

    #[test]
    fn history_clear_wipes_both_stacks() {
        let mut scene = Scene::default();
        let mut h = History::new(10);

        h.push(
            SceneCommand::InsertSource {
                idx: 0,
                src: src(1.0),
            },
            &mut scene,
        )
        .unwrap();
        h.undo(&mut scene).unwrap();
        assert!(h.can_redo());

        h.clear();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn history_last_action_name() {
        let mut scene = Scene::default();
        let mut h = History::new(10);
        assert_eq!(h.last_action_name(), None);

        h.push(
            SceneCommand::InsertSource {
                idx: 0,
                src: src(1.0),
            },
            &mut scene,
        )
        .unwrap();
        assert_eq!(h.last_action_name(), Some("Add source"));

        h.undo(&mut scene).unwrap();
        assert_eq!(h.last_action_name(), None);
        assert_eq!(h.next_redo_name(), Some("Add source"));
    }
}

use anyhow::{anyhow, Context, Result};
use rodio::{Decoder, OutputStream, Sink, Source};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::fs;
use crate::cli::colors::{EMERALD_BRIGHT, BLUE_BRIGHT, GRAY_DIM, RED_ERROR, RESET};
use crate::config::Config;

// Audio player with local playback and future SoundCloud integration
pub struct AudioPlayer {
    _stream: OutputStream,
    sink: Sink,
    current_track: Option<TrackInfo>,
    is_paused: Arc<AtomicBool>,
    speed: f32,
}

#[derive(Debug, Clone)]
pub struct TrackInfo {
    pub title: String,
    pub artist: String,
    pub duration: Option<Duration>,
    pub source: TrackSource,
}

#[derive(Debug, Clone)]
pub enum TrackSource {
    Local(PathBuf),
    SoundCloud { id: u64, stream_url: String },
}

#[derive(Debug, Clone)]
pub struct PlaybackStatus {
    pub is_playing: bool,
    pub is_paused: bool,
    pub current_track: Option<TrackInfo>,
    pub speed: f32,
    pub volume: f32,
}

impl AudioPlayer {
    pub fn new(_config: &Config) -> Result<Self> {
        let (_stream, stream_handle) = OutputStream::try_default()
            .context("Failed to initialize audio output stream")?;
        
        let sink = Sink::try_new(&stream_handle)
            .context("Failed to create audio sink")?;
        
        Ok(Self {
            _stream,
            sink,
            current_track: None,
            is_paused: Arc::new(AtomicBool::new(false)),
            speed: 1.0,
        })
    }

    /// Search for music locally or on SoundCloud
    pub async fn search_music(&self, query: &str) -> Result<Vec<TrackInfo>> {
        println!("{}🔍 Searching for: {}{}", BLUE_BRIGHT, query, RESET);
        
        let mut results = Vec::new();
        
        // Search SoundCloud first (faster and more comprehensive)
        let soundcloud_tracks = self.search_soundcloud(query).await.unwrap_or_default();
        results.extend(soundcloud_tracks);
        
        // Then search local files if no SoundCloud results
        if results.is_empty() {
            let local_tracks = self.search_local_music(query).await?;
            results.extend(local_tracks);
        }
        
        if results.is_empty() {
            println!("{}❌ No tracks found for: {}{}", RED_ERROR, query, RESET);
        } else {
            println!("{}✅ Found {} track(s){}", EMERALD_BRIGHT, results.len(), RESET);
        }
        
        Ok(results)
    }

    /// Search for local music files
    async fn search_local_music(&self, query: &str) -> Result<Vec<TrackInfo>> {
        let mut tracks = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Common music directories to search
        let music_dirs = [
            ".",
            "music",
            "songs", 
            "audio",
            &(std::env::var("USERPROFILE").unwrap_or_default() + "/Music"),
        ];
        
        for dir_str in &music_dirs {
            let dir = PathBuf::from(dir_str);
            if !dir.exists() {
                continue;
            }
            
            let found_tracks = self.scan_directory_for_music(&dir, &query_lower).await?;
            tracks.extend(found_tracks);
        }
        
        Ok(tracks)
    }

    /// Scan a directory for music files matching the query
    async fn scan_directory_for_music(&self, dir: &Path, query: &str) -> Result<Vec<TrackInfo>> {
        let mut tracks = Vec::new();
        
        if let Ok(mut entries) = fs::read_dir(dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                
                if path.is_file() {
                    if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
                        let ext_lower = extension.to_lowercase();
                        if matches!(ext_lower.as_str(), "mp3" | "wav" | "flac" | "ogg" | "m4a") {
                            let file_name = path.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_lowercase();
                            
                            if file_name.contains(query) {
                                let track = TrackInfo {
                                    title: path.file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("Unknown")
                                        .to_string(),
                                    artist: "Local File".to_string(),
                                    duration: None, // We could parse this from metadata
                                    source: TrackSource::Local(path),
                                };
                                tracks.push(track);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(tracks)
    }

    /// Search SoundCloud using their public API
    async fn search_soundcloud(&self, query: &str) -> Result<Vec<TrackInfo>> {
        let client = reqwest::Client::new();
        
        // SoundCloud public API endpoint (no auth required for basic search)
        let url = format!(
            "https://api-v2.soundcloud.com/search/tracks?q={}&limit=10",
            urlencoding::encode(query)
        );
        
        let response = client
            .get(&url)
            .header("User-Agent", "glimmer/1.0")
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Ok(Vec::new()); // Fail gracefully
        }
        
        let json: serde_json::Value = response.json().await?;
        let mut tracks = Vec::new();
        
        if let Some(collection) = json.get("collection").and_then(|c| c.as_array()) {
            for item in collection.iter().take(10) {
                if let (Some(title), Some(user), Some(id)) = (
                    item.get("title").and_then(|t| t.as_str()),
                    item.get("user").and_then(|u| u.get("username")).and_then(|un| un.as_str()),
                    item.get("id").and_then(|i| i.as_u64()),
                ) {
                    // Check if the track is streamable
                    let streamable = item.get("streamable").and_then(|s| s.as_bool()).unwrap_or(false);
                    
                    if streamable {
                        // Generate stream URL (SoundCloud format)
                        let stream_url = format!("https://api.soundcloud.com/tracks/{}/stream", id);
                        
                        let track = TrackInfo {
                            title: title.to_string(),
                            artist: user.to_string(),
                            duration: item.get("duration")
                                .and_then(|d| d.as_u64())
                                .map(|ms| Duration::from_millis(ms)),
                            source: TrackSource::SoundCloud {
                                id,
                                stream_url,
                            },
                        };
                        tracks.push(track);
                    }
                }
            }
        }
        
        if !tracks.is_empty() {
            println!("{}☁️  Found {} SoundCloud tracks{}", BLUE_BRIGHT, tracks.len(), RESET);
        }
        
        Ok(tracks)
    }

    /// Play a specific track
    pub async fn play_track(&mut self, track: &TrackInfo) -> Result<()> {
        println!("{}🎵 Playing: {} - {}{}", EMERALD_BRIGHT, track.artist, track.title, RESET);
        
        match &track.source {
            TrackSource::Local(path) => {
                self.play_local_file(path).await?;
            }
            TrackSource::SoundCloud { stream_url, .. } => {
                self.play_soundcloud_stream(stream_url).await?;
            }
        }
        
        self.current_track = Some(track.clone());
        self.is_paused.store(false, Ordering::Relaxed);
        
        Ok(())
    }

    /// Play a local audio file
    async fn play_local_file(&mut self, path: &Path) -> Result<()> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open audio file: {}", path.display()))?;
        
        let source = Decoder::new(file)
            .context("Failed to decode audio file")?;
        
        // Apply speed adjustment
        let source = source.speed(self.speed).amplify(if self.speed != 1.0 { 0.8 } else { 1.0 });
        
        self.sink.stop();
        self.sink.append(source);
        
        Ok(())
    }

    /// Play a SoundCloud stream
    async fn play_soundcloud_stream(&mut self, stream_url: &str) -> Result<()> {
        let response = reqwest::get(stream_url).await?;
        let bytes = response.bytes().await?;
        
        let cursor = Cursor::new(bytes);
        let source = Decoder::new(cursor)
            .context("Failed to decode SoundCloud stream")?;
        
        // Apply speed adjustment
        let source = source.speed(self.speed).amplify(if self.speed != 1.0 { 0.8 } else { 1.0 });
        
        self.sink.stop();
        self.sink.append(source);
        
        Ok(())
    }

    /// Set playback speed (0.5x to 3.0x)
    pub fn set_speed(&mut self, speed: f32) -> Result<()> {
        if speed < 0.25 || speed > 3.0 {
            return Err(anyhow!("Speed must be between 0.25x and 3.0x"));
        }
        
        self.speed = speed;
        println!("{}🏃 Playback speed set to {:.1}x{}", BLUE_BRIGHT, speed, RESET);
        
        // Note: Speed change will apply to the next track played
        // TODO: Implement real-time speed adjustment
        
        Ok(())
    }

    /// Pause/resume playback
    pub fn toggle_pause(&mut self) {
        if self.is_paused.load(Ordering::Relaxed) {
            self.sink.play();
            self.is_paused.store(false, Ordering::Relaxed);
            println!("{}▶️  Resumed playback{}", EMERALD_BRIGHT, RESET);
        } else {
            self.sink.pause();
            self.is_paused.store(true, Ordering::Relaxed);
            println!("{}⏸️  Paused playback{}", BLUE_BRIGHT, RESET);
        }
    }

    /// Stop playback
    pub fn stop(&mut self) {
        self.sink.stop();
        self.current_track = None;
        self.is_paused.store(false, Ordering::Relaxed);
        println!("{}⏹️  Stopped playback{}", GRAY_DIM, RESET);
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&mut self, volume: f32) {
        let clamped_volume = volume.clamp(0.0, 1.0);
        self.sink.set_volume(clamped_volume);
        println!("{}🔊 Volume set to {:.0}%{}", BLUE_BRIGHT, clamped_volume * 100.0, RESET);
    }

    /// Get current playback status
    pub fn get_status(&self) -> PlaybackStatus {
        PlaybackStatus {
            is_playing: !self.sink.empty() && !self.is_paused.load(Ordering::Relaxed),
            is_paused: self.is_paused.load(Ordering::Relaxed),
            current_track: self.current_track.clone(),
            speed: self.speed,
            volume: self.sink.volume(),
        }
    }

    /// Smart music search with artist recognition
    pub async fn smart_search(&self, query: &str) -> Result<Vec<TrackInfo>> {
        let query_lower = query.to_lowercase();
        
        // Parse queries like "play a charli xcx song" or "play blinding lights"
        let search_term = if query_lower.contains(" by ") {
            query.to_string()
        } else if query_lower.starts_with("play ") {
            query[5..].to_string()
        } else if query_lower.starts_with("a ") && query_lower.contains(" song") {
            // Extract artist name from "a [artist] song"
            let artist_part = &query_lower[2..];
            if let Some(song_idx) = artist_part.find(" song") {
                artist_part[..song_idx].to_string()
            } else {
                query.to_string()
            }
        } else {
            query.to_string()
        };
        
        println!("{}🎯 Smart search for: {}{}", BLUE_BRIGHT, search_term, RESET);
        self.search_music(&search_term).await
    }
}

/// Display search results in a nice format
pub fn display_search_results(tracks: &[TrackInfo]) {
    if tracks.is_empty() {
        return;
    }
    
    println!("\n{}🎵 Search Results:{}", EMERALD_BRIGHT, RESET);
    println!("{}─────────────────{}", GRAY_DIM, RESET);
    
    for (i, track) in tracks.iter().enumerate() {
        let source_icon = match &track.source {
            TrackSource::Local(_) => "📁",
            TrackSource::SoundCloud { .. } => "☁️",
        };
        
        println!("{}{}. {}{} {} - {}", 
            BLUE_BRIGHT, i + 1, RESET,
            source_icon,
            track.artist, 
            track.title
        );
    }
    
    println!();
    println!("{}💡 Use 'play <number>' to play a track{}", GRAY_DIM, RESET);
    println!("{}💡 Use 'speed <number>' to change playback speed (e.g., 'speed 1.5'){}", GRAY_DIM, RESET);
}
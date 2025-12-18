#pragma once

// Starts the audio UDP receiver and synchronized playback thread.
// Returns true if startup succeeded or the receiver was already running.
bool StartAudioReceiver();

// Stops the audio receiver and playback thread if running.
void StopAudioReceiver();

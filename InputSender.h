#ifndef INPUT_SENDER_H
#define INPUT_SENDER_H

#include <atomic>

// Main function for the input sender thread.
// It takes a reference to a boolean that controls its lifecycle.
void InputSendThread(std::atomic<bool>& running);

#endif // INPUT_SENDER_H

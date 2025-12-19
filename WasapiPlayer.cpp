#include "WasapiPlayer.h"
#include "DebugLog.h"
#include <comdef.h> // For _com_error

// --- COM Smart Pointer Helper ---
template<typename T>
void SafeRelease(T** ppT) {
    if (*ppT) {
        (*ppT)->Release();
        *ppT = nullptr;
    }
}

WasapiPlayer::WasapiPlayer(JitterBuffer& jitterBuffer) : m_jitterBuffer(jitterBuffer) {}

WasapiPlayer::~WasapiPlayer() {
    Stop();
}

bool WasapiPlayer::Start() {
    if (m_isRunning) {
        return true;
    }

    if (!InitializeWasapi()) {
        DebugLog(L"WasapiPlayer: Failed to initialize WASAPI.");
        ReleaseWasapi();
        return false;
    }

    m_isRunning = true;
    m_thread = std::thread(&WasapiPlayer::PlaybackThread, this);
    DebugLog(L"WasapiPlayer: Started.");
    return true;
}

void WasapiPlayer::Stop() {
    if (!m_isRunning) {
        return;
    }

    m_isRunning = false;
    if (m_hAudioEvent != nullptr) {
        SetEvent(m_hAudioEvent); // Wake up the thread so it can exit.
    }
    if (m_thread.joinable()) {
        m_thread.join();
    }

    ReleaseWasapi();
    DebugLog(L"WasapiPlayer: Stopped.");
}

bool WasapiPlayer::InitializeWasapi() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) return false;

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&m_pEnumerator);
    if (FAILED(hr)) return false;

    hr = m_pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &m_pDevice);
    if (FAILED(hr)) return false;

    hr = m_pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&m_pAudioClient);
    if (FAILED(hr)) return false;

    // The specification requires using a fixed format, not the device's mix format.
    m_waveFormat.wFormatTag = WAVE_FORMAT_PCM;
    m_waveFormat.nChannels = 2;
    m_waveFormat.nSamplesPerSec = 48000;
    m_waveFormat.wBitsPerSample = 16;
    m_waveFormat.nBlockAlign = (m_waveFormat.nChannels * m_waveFormat.wBitsPerSample) / 8;
    m_waveFormat.nAvgBytesPerSec = m_waveFormat.nSamplesPerSec * m_waveFormat.nBlockAlign;
    m_waveFormat.cbSize = 0;

    // Initialize the audio client with the protocol's fixed format.
    hr = m_pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 0, 0, &m_waveFormat, nullptr);
    if (FAILED(hr)) {
        DebugLog(L"WasapiPlayer: IAudioClient::Initialize failed with the fixed format. hr=0x%08X", hr);
        return false;
    }

    m_hAudioEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (m_hAudioEvent == nullptr) return false;

    hr = m_pAudioClient->SetEventHandle(m_hAudioEvent);
    if (FAILED(hr)) return false;

    hr = m_pAudioClient->GetBufferSize(&m_bufferFrameCount);
    if (FAILED(hr)) return false;

    hr = m_pAudioClient->GetService(__uuidof(IAudioRenderClient), (void**)&m_pRenderClient);
    if (FAILED(hr)) return false;

    return true;
}

void WasapiPlayer::ReleaseWasapi() {
    if (m_pAudioClient) {
        m_pAudioClient->Stop();
    }
    if (m_hAudioEvent) {
        CloseHandle(m_hAudioEvent);
        m_hAudioEvent = nullptr;
    }
    SafeRelease(&m_pRenderClient);
    SafeRelease(&m_pAudioClient);
    SafeRelease(&m_pDevice);
    SafeRelease(&m_pEnumerator);
    CoUninitialize();
}

void WasapiPlayer::PlaybackThread() {
    if (!m_pAudioClient || !m_pRenderClient) {
        return;
    }

    HRESULT hr = m_pAudioClient->Start();
    if(FAILED(hr)) {
        DebugLog(L"WasapiPlayer: Failed to start audio client. hr=0x%08X", hr);
        return;
    }

    while (m_isRunning) {
        DWORD waitResult = WaitForSingleObject(m_hAudioEvent, 200); // 200ms timeout
        if (!m_isRunning) {
            break;
        }

        if (waitResult == WAIT_OBJECT_0) {
            UINT32 padding;
            hr = m_pAudioClient->GetCurrentPadding(&padding);
            if (SUCCEEDED(hr)) {
                UINT32 framesAvailable = m_bufferFrameCount - padding;
                if (framesAvailable > 0) {
                    BYTE* pData;
                    hr = m_pRenderClient->GetBuffer(framesAvailable, &pData);
                    if (SUCCEEDED(hr)) {
                        // We need to provide 'framesAvailable' frames of audio.
                        // One frame = 2ch * 16bit = 4 bytes.
                        // One block = 1920 bytes = 480 frames.
                        // --- Synchronization Logic ---
                        if (!m_isSynchronized) {
                            if (m_jitterBuffer.PeekNextBlockId(m_expectedBlockId)) {
                                m_isSynchronized = true;
                                DebugLog(L"WasapiPlayer: Synchronized to BlockId %u.", m_expectedBlockId);
                            } else {
                                // Jitter buffer is empty, fill the entire WASAPI buffer with silence and wait.
                                memset(pData, 0, framesAvailable * m_waveFormat.nBlockAlign);
                                m_pRenderClient->ReleaseBuffer(framesAvailable, 0);
                                continue; // Go back to waiting on the event
                            }
                        }

                        // --- Safe Buffer Filling Logic ---
                        UINT32 bytesToWrite = framesAvailable * m_waveFormat.nBlockAlign;
                        UINT32 bytesWritten = 0;
                        BYTE* pBuffer = pData;

                        // 1. Use data from the partial buffer first.
                        if (!m_partialBlockBuffer.empty()) {
                            size_t bytesToCopy = std::min((size_t)bytesToWrite, m_partialBlockBuffer.size());
                            memcpy(pBuffer, m_partialBlockBuffer.data(), bytesToCopy);
                            bytesWritten += bytesToCopy;
                            pBuffer += bytesToCopy;
                            // Remove the copied data from the partial buffer
                            m_partialBlockBuffer.erase(m_partialBlockBuffer.begin(), m_partialBlockBuffer.begin() + bytesToCopy);
                        }

                        // 2. Fill the rest of the buffer with new blocks.
                        while (bytesWritten < bytesToWrite) {
                            std::vector<uint8_t> pcmData;
                            UINT32 bytesRemaining = bytesToWrite - bytesWritten;

                            if (m_jitterBuffer.GetNextBlock(pcmData, m_expectedBlockId)) {
                                // We got a block. It might be larger than the space left in the buffer.
                                size_t bytesToCopy = std::min((size_t)bytesRemaining, pcmData.size());
                                memcpy(pBuffer, pcmData.data(), bytesToCopy);
                                bytesWritten += bytesToCopy;
                                pBuffer += bytesToCopy;

                                // If we didn't use the whole block, save the rest.
                                if (bytesToCopy < pcmData.size()) {
                                    m_partialBlockBuffer.assign(pcmData.begin() + bytesToCopy, pcmData.end());
                                }
                            } else {
                                // Block is missing, insert silence for the size of one block.
                                DebugLog(L"WasapiPlayer: Block %u is missing, inserting silence.", m_expectedBlockId);
                                size_t silenceBytes = 1920; // Size of one audio block in S16LE.
                                size_t bytesToCopy = std::min((size_t)bytesRemaining, silenceBytes);
                                memset(pBuffer, 0, bytesToCopy);
                                bytesWritten += bytesToCopy;
                                pBuffer += bytesToCopy;
                            }
                            m_expectedBlockId++;
                        }

                        m_pRenderClient->ReleaseBuffer(framesAvailable, 0);
                    }
                }
            }
        }
    }

    m_pAudioClient->Stop();
}

#include "AudioReceiver.h"
#include "DebugLog.h"
#include "Globals.h"
#include "ReedSolomon.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <cmath>
#include <map>
#include <mutex>
#include <vector>
#include <queue>
#include <chrono>
#include <condition_variable>
#include <atlbase.h>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <array>

#pragma comment(lib, "ole32.lib")

namespace
{
    constexpr uint32_t AUDIO_SAMPLE_RATE = 48000;
    constexpr uint16_t AUDIO_CHANNELS = 2;
    constexpr uint16_t AUDIO_BITS_PER_SAMPLE = 16;
    constexpr uint32_t PCM_BYTES_PER_BLOCK = 1920;
    constexpr uint32_t BLOCK_SAMPLES_PER_CH = 480;
    constexpr uint32_t BLOCK_TIMEOUT_MS = 15;
    constexpr uint32_t JITTER_BLOCKS = 1;
    constexpr uint16_t AUDIO_PORT = 8200;

    constexpr uint8_t FLAG_IS_PARITY = 0x01;
    constexpr uint8_t FLAG_BLOCK_START = 0x02;
    constexpr uint8_t FLAG_BLOCK_END = 0x04;

    #pragma pack(push, 1)
    struct AudioPacketHeader
    {
        char     Magic[4];
        uint8_t  Version;
        uint8_t  HeaderSize;
        uint8_t  Flags;
        uint8_t  Reserved0;
        uint32_t SessionId;
        uint32_t BlockId;
        uint16_t ShardIndex;
        uint16_t ShardTotal;
        uint16_t K;
        uint16_t M;
        uint16_t ShardBytes;
        uint16_t PcmBytesInBlock;
        uint16_t BlockSamplesPerCh;
        uint16_t Reserved1;
        uint32_t SampleRate;
        uint16_t Channels;
        uint16_t BitsPerSample;
        uint64_t CaptureTimestampNs;
        uint32_t BlockCrc32;
        uint32_t PayloadCrc32;
        uint32_t HeaderCrc32;
        uint32_t Reserved2;
    };
    #pragma pack(pop)
    static_assert(sizeof(AudioPacketHeader) == 64, "AudioPacketHeader must be 64 bytes");

    uint32_t Crc32(const uint8_t* data, size_t len)
    {
        static uint32_t table[256];
        static std::once_flag initFlag;
        std::call_once(initFlag, [] {
            for (uint32_t i = 0; i < 256; ++i)
            {
                uint32_t crc = i;
                for (int j = 0; j < 8; ++j)
                {
                    if (crc & 1)
                        crc = (crc >> 1) ^ 0xEDB88320;
                    else
                        crc >>= 1;
                }
                table[i] = crc;
            }
        });

        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < len; ++i)
        {
            uint8_t byte = data[i];
            uint32_t idx = (crc ^ byte) & 0xFF;
            crc = (crc >> 8) ^ table[idx];
        }
        return crc ^ 0xFFFFFFFF;
    }

    std::wstring HexDump(const std::vector<uint8_t>& bytes)
    {
        std::wstringstream ss;
        ss << std::hex << std::uppercase << std::setfill(L'0');
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            ss << std::setw(2) << static_cast<int>(bytes[i]);
            if (i + 1 != bytes.size()) ss << L" ";
        }
        return ss.str();
    }

    struct BlockKey
    {
        uint32_t sessionId;
        uint32_t blockId;
        bool operator==(const BlockKey& other) const noexcept
        {
            return sessionId == other.sessionId && blockId == other.blockId;
        }
    };

    struct BlockKeyHash
    {
        size_t operator()(const BlockKey& key) const noexcept
        {
            return (static_cast<size_t>(key.sessionId) << 32) ^ key.blockId;
        }
    };

    struct BlockState
    {
        uint16_t k = 0;
        uint16_t m = 0;
        uint16_t shardBytes = 0;
        uint16_t shardTotal = 0;
        uint32_t pcmBytes = 0;
        uint64_t captureTimestamp = 0;
        uint32_t blockCrc32 = 0;
        std::vector<std::vector<uint8_t>> dataShards;
        std::vector<std::vector<uint8_t>> parityShards;
        uint16_t received = 0;
    };

    struct DecodedBlock
    {
        uint32_t sessionId = 0;
        uint32_t blockId = 0;
        uint64_t captureTimestamp = 0;
        uint16_t shardBytes = 0;
        uint16_t shardTotal = 0;
        uint16_t k = 0;
        uint16_t m = 0;
        std::vector<uint8_t> pcm;
    };

    std::atomic<bool> g_audioReceiverRunning{true};
    std::atomic<bool> g_audioPlaybackRunning{true};
    std::thread g_audioReceiverThread;
    std::thread g_audioPlaybackThread;

    std::mutex g_blockMutex;
    std::unordered_map<BlockKey, BlockState, BlockKeyHash> g_blocks;
    uint32_t g_currentSession = 0;
    uint32_t g_expectedBlock = 0;

    std::mutex g_jitterMutex;
    std::condition_variable g_jitterCv;
    std::map<uint32_t, DecodedBlock> g_jitterBuffer;
    uint32_t g_playSession = 0;

    enum class DropReason
    {
        SizeTooSmall = 0,
        MagicMismatch,
        VersionMismatch,
        SizeMismatch,
        HeaderCrcMismatch,
        PayloadCrcMismatch,
        ShardTotalMismatch,
        ShardIndexRange,
        PcmBytesMismatch,
        SamplesPerChannelMismatch,
        SampleRateMismatch,
        ChannelFormatMismatch,
        ShardBytesZero,
        Count
    };

    const wchar_t* DropReasonName(DropReason reason)
    {
        switch (reason)
        {
        case DropReason::SizeTooSmall: return L"size_too_small";
        case DropReason::MagicMismatch: return L"magic_mismatch";
        case DropReason::VersionMismatch: return L"version_or_headersize";
        case DropReason::SizeMismatch: return L"size_mismatch";
        case DropReason::HeaderCrcMismatch: return L"header_crc_mismatch";
        case DropReason::PayloadCrcMismatch: return L"payload_crc_mismatch";
        case DropReason::ShardTotalMismatch: return L"shard_total_mismatch";
        case DropReason::ShardIndexRange: return L"shard_index_range";
        case DropReason::PcmBytesMismatch: return L"pcm_bytes_mismatch";
        case DropReason::SamplesPerChannelMismatch: return L"samples_per_ch_mismatch";
        case DropReason::SampleRateMismatch: return L"sample_rate_mismatch";
        case DropReason::ChannelFormatMismatch: return L"channel_format_mismatch";
        case DropReason::ShardBytesZero: return L"shard_bytes_zero";
        default: return L"unknown";
        }
    }

    void LogValidationDrop(DropReason reason)
    {
        static std::array<uint64_t, static_cast<size_t>(DropReason::Count)> counts{};
        static auto lastLog = std::chrono::steady_clock::now();

        const size_t idx = static_cast<size_t>(reason);
        if (idx < counts.size())
        {
            ++counts[idx];
        }

        auto now = std::chrono::steady_clock::now();
        if (now - lastLog >= std::chrono::seconds(1))
        {
            std::wstringstream ss;
            ss << L"AudioReceiver: dropped packets - ";
            bool any = false;
            for (size_t i = 0; i < counts.size(); ++i)
            {
                if (counts[i] == 0) continue;
                if (any) ss << L", ";
                ss << DropReasonName(static_cast<DropReason>(i)) << L":" << counts[i];
                any = true;
            }
            if (any)
            {
                DebugLog(ss.str());
            }
            counts.fill(0);
            lastLog = now;
        }
    }

    void ResetForSession(uint32_t session)
    {
        std::scoped_lock lk(g_blockMutex, g_jitterMutex);
        g_blocks.clear();
        g_jitterBuffer.clear();
        g_currentSession = session;
        g_playSession = session;
        g_expectedBlock = 0;
    }

    bool ValidateAndParse(const uint8_t* buf, size_t len, AudioPacketHeader& header)
    {
        if (len < sizeof(AudioPacketHeader))
        {
            LogValidationDrop(DropReason::SizeTooSmall);
            return false;
        }

        std::memcpy(&header, buf, sizeof(AudioPacketHeader));
        if (std::memcmp(header.Magic, "AUDP", 4) != 0)
        {
            LogValidationDrop(DropReason::MagicMismatch);
            return false;
        }
        if (header.Version != 1 || header.HeaderSize != sizeof(AudioPacketHeader))
        {
            LogValidationDrop(DropReason::VersionMismatch);
            return false;
        }
        if (header.ShardBytes == 0)
        {
            LogValidationDrop(DropReason::ShardBytesZero);
            return false;
        }

        const size_t expectedSize = sizeof(AudioPacketHeader) + header.ShardBytes;
        if (len != expectedSize)
        {
            LogValidationDrop(DropReason::SizeMismatch);
            return false;
        }

        AudioPacketHeader crcHeader = header;
        crcHeader.HeaderCrc32 = 0;
        const uint32_t headerCrc = Crc32(reinterpret_cast<uint8_t*>(&crcHeader), sizeof(AudioPacketHeader));
        if (headerCrc != header.HeaderCrc32)
        {
            LogValidationDrop(DropReason::HeaderCrcMismatch);
            return false;
        }

        const uint8_t* payload = buf + sizeof(AudioPacketHeader);
        const uint32_t payloadCrc = Crc32(payload, header.ShardBytes);
        if (payloadCrc != header.PayloadCrc32)
        {
            LogValidationDrop(DropReason::PayloadCrcMismatch);
            return false;
        }
        if (header.ShardTotal != header.K + header.M)
        {
            LogValidationDrop(DropReason::ShardTotalMismatch);
            return false;
        }
        if (header.ShardIndex >= header.ShardTotal)
        {
            LogValidationDrop(DropReason::ShardIndexRange);
            return false;
        }
        if (header.PcmBytesInBlock != PCM_BYTES_PER_BLOCK)
        {
            LogValidationDrop(DropReason::PcmBytesMismatch);
            return false;
        }
        if (header.BlockSamplesPerCh != BLOCK_SAMPLES_PER_CH)
        {
            LogValidationDrop(DropReason::SamplesPerChannelMismatch);
            return false;
        }
        if (header.SampleRate != AUDIO_SAMPLE_RATE)
        {
            LogValidationDrop(DropReason::SampleRateMismatch);
            return false;
        }
        if (header.Channels != AUDIO_CHANNELS || header.BitsPerSample != AUDIO_BITS_PER_SAMPLE)
        {
            LogValidationDrop(DropReason::ChannelFormatMismatch);
            return false;
        }
        return true;
    }

    void LogDecodedBlock(const DecodedBlock& blk)
    {
        AudioPacketHeader header{};
        std::memcpy(header.Magic, "AUDP", 4);
        header.Version = 1;
        header.HeaderSize = sizeof(AudioPacketHeader);
        header.Flags = FLAG_BLOCK_START;
        header.SessionId = blk.sessionId;
        header.BlockId = blk.blockId;
        header.ShardIndex = 0;
        header.ShardTotal = blk.shardTotal;
        header.K = blk.k;
        header.M = blk.m;
        header.ShardBytes = blk.shardBytes;
        header.PcmBytesInBlock = PCM_BYTES_PER_BLOCK;
        header.BlockSamplesPerCh = BLOCK_SAMPLES_PER_CH;
        header.SampleRate = AUDIO_SAMPLE_RATE;
        header.Channels = AUDIO_CHANNELS;
        header.BitsPerSample = AUDIO_BITS_PER_SAMPLE;
        header.CaptureTimestampNs = blk.captureTimestamp;
        header.BlockCrc32 = Crc32(blk.pcm.data(), PCM_BYTES_PER_BLOCK);
        header.PayloadCrc32 = header.BlockCrc32;
        header.Reserved0 = header.Reserved1 = header.Reserved2 = 0;

        AudioPacketHeader crcHeader = header;
        crcHeader.HeaderCrc32 = 0;
        header.HeaderCrc32 = Crc32(reinterpret_cast<uint8_t*>(&crcHeader), sizeof(AudioPacketHeader));

        std::vector<uint8_t> dump(sizeof(AudioPacketHeader));
        std::memcpy(dump.data(), &header, sizeof(AudioPacketHeader));
        dump.insert(dump.end(), blk.pcm.begin(), blk.pcm.begin() + PCM_BYTES_PER_BLOCK);
        DebugLog(L"AudioBlockDecoded: " + HexDump(dump));
    }

    void TryAssemble(const AudioPacketHeader& header, const uint8_t* payload)
    {
        BlockKey key{header.SessionId, header.BlockId};
        std::unique_lock<std::mutex> lock(g_blockMutex);
        if (g_currentSession != 0 && g_currentSession != header.SessionId)
        {
            lock.unlock();
            ResetForSession(header.SessionId);
            lock.lock();
        }
        if (g_currentSession == 0) g_currentSession = header.SessionId;

        auto& st = g_blocks[key];
        if (st.k == 0)
        {
            st.k = header.K;
            st.m = header.M;
            st.shardBytes = header.ShardBytes;
            st.shardTotal = header.ShardTotal;
            st.pcmBytes = header.PcmBytesInBlock;
            st.captureTimestamp = header.CaptureTimestampNs;
            st.blockCrc32 = header.BlockCrc32;
            st.dataShards.assign(st.k, {});
            st.parityShards.assign(st.m, {});
        }

        if (header.ShardIndex < st.k)
        {
            if (st.dataShards[header.ShardIndex].empty())
            {
                st.dataShards[header.ShardIndex] = std::vector<uint8_t>(payload, payload + header.ShardBytes);
                ++st.received;
            }
        }
        else
        {
            const uint16_t parityIdx = header.ShardIndex - st.k;
            if (parityIdx < st.m && st.parityShards[parityIdx].empty())
            {
                st.parityShards[parityIdx] = std::vector<uint8_t>(payload, payload + header.ShardBytes);
                ++st.received;
            }
        }

        if (st.received < st.k)
        {
            return;
        }

        std::map<uint32_t, std::vector<uint8_t>> shardMap;
        for (uint16_t i = 0; i < st.k; ++i)
        {
            if (!st.dataShards[i].empty()) shardMap.emplace(i, st.dataShards[i]);
        }
        for (uint16_t i = 0; i < st.m; ++i)
        {
            if (!st.parityShards[i].empty()) shardMap.emplace(st.k + i, st.parityShards[i]);
        }

        lock.unlock();

        std::vector<uint8_t> decoded;
        if (!DecodeFEC_ISAL(shardMap, st.k, st.m, st.pcmBytes, decoded))
        {
            return;
        }
        if (decoded.size() < st.pcmBytes) return;
        decoded.resize(st.pcmBytes);
        const uint32_t crc = Crc32(decoded.data(), st.pcmBytes);
        if (crc != st.blockCrc32) return;

        DecodedBlock blk;
        blk.sessionId = header.SessionId;
        blk.blockId = header.BlockId;
        blk.captureTimestamp = st.captureTimestamp;
        blk.shardBytes = st.shardBytes;
        blk.shardTotal = st.shardTotal;
        blk.k = st.k;
        blk.m = st.m;
        blk.pcm = std::move(decoded);

        {
            std::lock_guard<std::mutex> mapLock(g_blockMutex);
            g_blocks.erase(key);
        }

        {
            std::lock_guard<std::mutex> jitterLock(g_jitterMutex);
            if (g_playSession != blk.sessionId)
            {
                g_jitterBuffer.clear();
                g_expectedBlock = blk.blockId;
                g_playSession = blk.sessionId;
            }
            auto [it, inserted] = g_jitterBuffer.emplace(header.BlockId, std::move(blk));
            if (inserted)
            {
                LogDecodedBlock(it->second);
            }
        }
        g_jitterCv.notify_one();
    }

    void ReceiverThread()
    {
        SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCKET)
        {
            DebugLog(L"AudioReceiver: socket failed");
            return;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(AUDIO_PORT);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR)
        {
            DebugLog(L"AudioReceiver: bind failed");
            closesocket(sock);
            return;
        }

        u_long nonBlocking = 1;
        if (ioctlsocket(sock, FIONBIO, &nonBlocking) == SOCKET_ERROR)
        {
            DebugLog(L"AudioReceiver: ioctlsocket(FIONBIO) failed");
        }

        std::vector<uint8_t> buffer;
        buffer.resize(2048);
        while (g_audioReceiverRunning.load(std::memory_order_relaxed))
        {
            int ret = recv(sock, reinterpret_cast<char*>(buffer.data()), static_cast<int>(buffer.size()), 0);
            if (ret <= 0)
            {
                if (WSAGetLastError() == WSAEWOULDBLOCK)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                break;
            }
            AudioPacketHeader header{};
            if (!ValidateAndParse(buffer.data(), static_cast<size_t>(ret), header))
            {
                continue;
            }
            TryAssemble(header, buffer.data() + sizeof(AudioPacketHeader));
        }
        closesocket(sock);
    }

    void PlaybackThread()
    {
        const HRESULT coHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        const bool coInitialized = SUCCEEDED(coHr);
        if (!coInitialized)
        {
            DebugLog(L"AudioPlayback: CoInitializeEx failed");
            return;
        }
        CComPtr<IMMDeviceEnumerator> enumerator;
        if (FAILED(enumerator.CoCreateInstance(__uuidof(MMDeviceEnumerator))))
        {
            DebugLog(L"AudioPlayback: MMDeviceEnumerator creation failed");
            CoUninitialize();
            return;
        }
        CComPtr<IMMDevice> device;
        if (FAILED(enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device)))
        {
            DebugLog(L"AudioPlayback: GetDefaultAudioEndpoint failed");
            CoUninitialize();
            return;
        }
        CComPtr<IAudioClient> client;
        if (FAILED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client))))
        {
            DebugLog(L"AudioPlayback: Activate IAudioClient failed");
            CoUninitialize();
            return;
        }

        WAVEFORMATEX format{};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = AUDIO_CHANNELS;
        format.nSamplesPerSec = AUDIO_SAMPLE_RATE;
        format.wBitsPerSample = AUDIO_BITS_PER_SAMPLE;
        format.nBlockAlign = (format.nChannels * format.wBitsPerSample) / 8;
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
        format.cbSize = 0;

        REFERENCE_TIME hnsBufferDuration = 20 * 10'000; // 20ms
        if (FAILED(client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, hnsBufferDuration, 0, &format, nullptr)))
        {
            DebugLog(L"AudioPlayback: Initialize failed");
            CoUninitialize();
            return;
        }

        HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!eventHandle)
        {
            DebugLog(L"AudioPlayback: CreateEvent failed");
            CoUninitialize();
            return;
        }
        client->SetEventHandle(eventHandle);

        CComPtr<IAudioRenderClient> renderClient;
        if (FAILED(client->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&renderClient))))
        {
            DebugLog(L"AudioPlayback: GetService failed");
            CloseHandle(eventHandle);
            CoUninitialize();
            return;
        }

        UINT32 bufferFrameCount = 0;
        client->GetBufferSize(&bufferFrameCount);
        client->Start();

        bool primed = false;
        uint64_t silenceBlocks = 0;
        while (g_audioPlaybackRunning.load(std::memory_order_relaxed))
        {
            WaitForSingleObject(eventHandle, 10);
            UINT32 padding = 0;
            if (FAILED(client->GetCurrentPadding(&padding)))
            {
                break;
            }
            UINT32 framesAvailable = bufferFrameCount - padding;
            while (framesAvailable >= BLOCK_SAMPLES_PER_CH)
            {
                DecodedBlock block;
                bool haveBlock = false;
                bool generatedSilence = false;
                {
                    std::unique_lock<std::mutex> lock(g_jitterMutex);
                    g_jitterCv.wait_for(lock, std::chrono::milliseconds(BLOCK_TIMEOUT_MS), [&] {
                        return !g_audioPlaybackRunning.load(std::memory_order_relaxed) || !g_jitterBuffer.empty();
                    });

                    if (!primed)
                    {
                        primed = true;
                        if (!g_jitterBuffer.empty())
                        {
                            g_expectedBlock = g_jitterBuffer.begin()->first;
                            g_playSession = g_jitterBuffer.begin()->second.sessionId;
                        }
                        else if (g_currentSession != 0)
                        {
                            g_playSession = g_currentSession;
                        }
                    }

                    while (!g_jitterBuffer.empty() && g_jitterBuffer.begin()->first < g_expectedBlock)
                    {
                        g_jitterBuffer.erase(g_jitterBuffer.begin());
                    }

                    if (primed)
                    {
                        auto it = g_jitterBuffer.find(g_expectedBlock);
                        if (it != g_jitterBuffer.end())
                        {
                            block = std::move(it->second);
                            g_jitterBuffer.erase(it);
                            haveBlock = true;
                        }
                        else
                        {
                            block.pcm.assign(PCM_BYTES_PER_BLOCK, 0);
                            block.sessionId = g_playSession;
                            block.blockId = g_expectedBlock;
                            block.shardBytes = PCM_BYTES_PER_BLOCK;
                            haveBlock = true;
                            generatedSilence = true;
                        }
                    }
                }

                if (!primed || !haveBlock)
                {
                    break;
                }

                BYTE* data = nullptr;
                if (FAILED(renderClient->GetBuffer(BLOCK_SAMPLES_PER_CH, &data)))
                {
                    g_audioPlaybackRunning.store(false, std::memory_order_relaxed);
                    break;
                }
                std::memcpy(data, block.pcm.data(), PCM_BYTES_PER_BLOCK);
                renderClient->ReleaseBuffer(BLOCK_SAMPLES_PER_CH, 0);
                framesAvailable -= BLOCK_SAMPLES_PER_CH;
                ++g_expectedBlock;

                if (generatedSilence)
                {
                    ++silenceBlocks;
                    if (silenceBlocks % 100 == 0)
                    {
                        std::wstringstream ss;
                        ss << L"AudioPlayback: generated " << silenceBlocks << L" silent blocks so far";
                        DebugLog(ss.str());
                    }
                }
            }
        }
        client->Stop();
        CloseHandle(eventHandle);
        if (coInitialized) CoUninitialize();
    }
}

namespace Audio
{
    void StartAudioPipeline()
    {
        g_audioReceiverRunning.store(true, std::memory_order_relaxed);
        g_audioPlaybackRunning.store(true, std::memory_order_relaxed);
        g_audioReceiverThread = std::thread(ReceiverThread);
        g_audioPlaybackThread = std::thread(PlaybackThread);
    }

    void StopAudioPipeline()
    {
        g_audioReceiverRunning.store(false, std::memory_order_relaxed);
        g_audioPlaybackRunning.store(false, std::memory_order_relaxed);
        g_jitterCv.notify_all();
        if (g_audioReceiverThread.joinable()) g_audioReceiverThread.join();
        if (g_audioPlaybackThread.joinable()) g_audioPlaybackThread.join();
    }
}

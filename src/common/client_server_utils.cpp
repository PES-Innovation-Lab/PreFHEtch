#include <client_server_utils.h>

void Timer::StartTimer() {
    m_TimerStart = std::chrono::high_resolution_clock::now();
}

void Timer::StopTimer() {
    m_TimerEnd = std::chrono::high_resolution_clock::now();
}

long long Timer::getDurationMilliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_TimerEnd -
                                                                 m_TimerStart)
        .count();
}

std::size_t getSizeInMB(const std::size_t &size_bytes) {
    return size_bytes / 1000000;
}

#include <client_server_utils.h>

void Timer::StartTimer() {
    m_TimerStart = std::chrono::high_resolution_clock::now();
}

void Timer::StopTimer() {
    m_TimerEnd = std::chrono::high_resolution_clock::now();
}

long long Timer::getDurationMicroseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(m_TimerEnd -
                                                                 m_TimerStart)
        .count();
}

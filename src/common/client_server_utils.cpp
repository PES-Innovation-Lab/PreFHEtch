#include <client_server_utils.h>

void Timer::StartTimer() {
    m_TimerStart = std::chrono::high_resolution_clock::now();
}

void Timer::StopTimer() {
    m_TimerEnd = std::chrono::high_resolution_clock::now();
}

void Timer::getDuration(long long &time_micro, long long &time_milli) const {
    auto start_time =
        std::chrono::time_point_cast<std::chrono::microseconds>(m_TimerStart)
            .time_since_epoch()
            .count();

    auto end_time =
        std::chrono::time_point_cast<std::chrono::microseconds>(m_TimerEnd)
            .time_since_epoch()
            .count();

    time_micro = end_time - start_time;
    time_milli = time_micro * 0.001;
}

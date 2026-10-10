// Statically linked FFmpeg backend for the MinGW x86 build. It decodes MP4
// frames to the D3D11 texture and AAC to PCM for waveOut, so Wine does not
// need a Media Foundation codec, DirectShow filter, LAV, or GStreamer.
namespace ffmpeg_win {

static AVFormatContext* s_fmt = NULL;
static AVCodecContext* s_vctx = NULL;
static AVCodecContext* s_actx = NULL;
static SwrContext* s_swr = NULL;
static SwsContext* s_sws = NULL;
static AVPacket* s_pkt = NULL;
static AVFrame* s_frame = NULL;
static int s_vstream = -1;
static int s_astream = -1;
static int s_width = 0;
static int s_height = 0;
static double s_fps = 30.0;
static int s_frameIndex = -1;
static int s_baseFrame = 0;
static int s_playToFrame = 0;
static int64_t s_tsOrigin = AV_NOPTS_VALUE;
static DWORD s_startTicks = 0;
static BOOL s_active = FALSE;
static BOOL s_endEvent = FALSE;
static BOOL s_eof = FALSE;
static BOOL s_flushed = FALSE;
static BOOL s_loggedFirstPresent = FALSE;
static unsigned char* s_rgba = NULL;
static int s_pitch = 0;
static MarniHandle s_tex = MARNI_NULL_HANDLE;
static int s_texW = 0;
static int s_texH = 0;
static short* s_audio = NULL;
static long s_audioFrames = 0;
static int s_audioRate = 22050;
static long s_audioStart = 0;
static long s_audioEnd = 0;
static BOOL s_useAudio = FALSE;
static HWAVEOUT s_wave = NULL;
static WAVEHDR s_hdr = {};
static BOOL s_hdrPrepared = FALSE;

static void ReleaseAudio(void)
{
    if (s_wave != NULL) {
        waveOutReset(s_wave);
        if (s_hdrPrepared) {
            waveOutUnprepareHeader(s_wave, &s_hdr, sizeof(s_hdr));
            s_hdrPrepared = FALSE;
        }
        waveOutClose(s_wave);
        s_wave = NULL;
    }
    ZeroMemory(&s_hdr, sizeof(s_hdr));
    free(s_audio);
    s_audio = NULL;
    s_audioFrames = 0;
    s_useAudio = FALSE;
}

static void Release(void)
{
    ReleaseAudio();
    if (s_swr) swr_free(&s_swr);
    if (s_sws) sws_freeContext(s_sws), s_sws = NULL;
    if (s_frame) av_frame_free(&s_frame);
    if (s_pkt) av_packet_free(&s_pkt);
    if (s_vctx) avcodec_free_context(&s_vctx);
    if (s_actx) avcodec_free_context(&s_actx);
    if (s_fmt) avformat_close_input(&s_fmt);
    free(s_rgba);
    s_rgba = NULL;
    s_pitch = 0;
    if (s_tex != MARNI_NULL_HANDLE) {
        MarniDX* dx = Marni_DX();
        if (dx) dx->DestroyTexture(s_tex);
        s_tex = MARNI_NULL_HANDLE;
    }
    s_texW = s_texH = 0;
    s_vstream = s_astream = -1;
    s_width = s_height = 0;
    s_fps = 30.0;
    s_frameIndex = -1;
    s_baseFrame = s_playToFrame = 0;
    s_tsOrigin = AV_NOPTS_VALUE;
    s_active = s_endEvent = s_eof = s_flushed = FALSE;
    s_loggedFirstPresent = FALSE;
}

static BOOL DecodeAudio(void)
{
    if (!s_actx || s_astream < 0 || s_actx->sample_rate <= 0) return FALSE;

    AVChannelLayout outLayout = AV_CHANNEL_LAYOUT_STEREO;
    AVChannelLayout inLayout = {};
    if (s_actx->ch_layout.nb_channels > 0)
        av_channel_layout_copy(&inLayout, &s_actx->ch_layout);
    else
        av_channel_layout_default(&inLayout, 2);
    int ret = swr_alloc_set_opts2(&s_swr, &outLayout, AV_SAMPLE_FMT_S16,
                                  s_audioRate, &inLayout, s_actx->sample_fmt,
                                  s_actx->sample_rate, 0, NULL);
    av_channel_layout_uninit(&inLayout);
    if (ret < 0 || !s_swr || swr_init(s_swr) < 0) return FALSE;

    AVFrame* frame = av_frame_alloc();
    if (!frame) return FALSE;
    long capacity = 0;
    long written = 0;
    BOOL ok = TRUE;

    while (ok && av_read_frame(s_fmt, s_pkt) >= 0) {
        if (s_pkt->stream_index != s_astream) {
            av_packet_unref(s_pkt);
            continue;
        }
        ret = avcodec_send_packet(s_actx, s_pkt);
        av_packet_unref(s_pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        for (;;) {
            ret = avcodec_receive_frame(s_actx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) { ok = FALSE; break; }

            const int maxOut = (int)av_rescale_rnd(
                swr_get_delay(s_swr, s_actx->sample_rate) + frame->nb_samples,
                s_audioRate, s_actx->sample_rate, AV_ROUND_UP);
            if (written + maxOut + 1 > capacity) {
                long want = written + maxOut + 4096;
                short* grown = (short*)realloc(
                    s_audio, (size_t)want * 2 * sizeof(short));
                if (!grown) { ok = FALSE; break; }
                s_audio = grown;
                capacity = want;
            }
            uint8_t* output[1] = {
                (uint8_t*)(s_audio + (size_t)written * 2)
            };
            int got = swr_convert(s_swr, output, maxOut,
                (const uint8_t**)frame->extended_data, frame->nb_samples);
            if (got < 0) { ok = FALSE; break; }
            written += got;
            av_frame_unref(frame);
        }
    }

    // Drain delayed AAC frames, then the resampler's tail.
    if (ok) {
        avcodec_send_packet(s_actx, NULL);
        while (avcodec_receive_frame(s_actx, frame) == 0) {
            const int maxOut = (int)av_rescale_rnd(
                swr_get_delay(s_swr, s_actx->sample_rate) + frame->nb_samples,
                s_audioRate, s_actx->sample_rate, AV_ROUND_UP);
            if (written + maxOut + 4096 > capacity) {
                long want = written + maxOut + 4096;
                short* grown = (short*)realloc(
                    s_audio, (size_t)want * 2 * sizeof(short));
                if (!grown) { ok = FALSE; break; }
                s_audio = grown;
                capacity = want;
            }
            uint8_t* output[1] = {
                (uint8_t*)(s_audio + (size_t)written * 2)
            };
            int got = swr_convert(s_swr, output, maxOut,
                (const uint8_t**)frame->extended_data, frame->nb_samples);
            if (got > 0) written += got;
            av_frame_unref(frame);
        }
        while (ok && written + 4096 <= capacity) {
            uint8_t* output[1] = {
                (uint8_t*)(s_audio + (size_t)written * 2)
            };
            int got = swr_convert(s_swr, output, 4096, NULL, 0);
            if (got <= 0) break;
            written += got;
        }
    }

    av_frame_free(&frame);
    if (!ok) { free(s_audio); s_audio = NULL; return FALSE; }
    s_audioFrames = written;
    s_useAudio = written > 0;

    // Reset the decoder and demuxer for video playback.
    avcodec_flush_buffers(s_actx);
    av_seek_frame(s_fmt, s_vstream, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(s_vctx);

    if (s_useAudio) {
        WAVEFORMATEX wfx = {};
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.nChannels = 2;
        wfx.nSamplesPerSec = (DWORD)s_audioRate;
        wfx.wBitsPerSample = 16;
        wfx.nBlockAlign = 4;
        wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
        if (waveOutOpen(&s_wave, WAVE_MAPPER, &wfx, 0, 0, CALLBACK_NULL) !=
            MMSYSERR_NOERROR) {
            s_wave = NULL;
            s_useAudio = FALSE;
            dbg_printf("[VIDEO] FFmpeg AAC decoded (%ld frames) but waveOut unavailable; continuing video silently\n",
                       s_audioFrames);
        } else {
            dbg_printf("[VIDEO] FFmpeg AAC audio ready: %ld stereo frames at %d Hz\n",
                       s_audioFrames, s_audioRate);
        }
    }
    return s_useAudio;
}

static BOOL ConvertFrame(void)
{
    if (!s_frame || s_width <= 0 || s_height <= 0) return FALSE;
    if (!s_rgba) {
        s_pitch = s_width * 4;
        s_rgba = (unsigned char*)malloc((size_t)s_pitch * s_height);
        if (!s_rgba) return FALSE;
    }
    s_sws = sws_getCachedContext(s_sws, s_width, s_height,
        (AVPixelFormat)s_frame->format, s_width, s_height, AV_PIX_FMT_RGBA,
        SWS_BILINEAR, NULL, NULL, NULL);
    if (!s_sws) return FALSE;
    uint8_t* dst[4] = { s_rgba, NULL, NULL, NULL };
    int stride[4] = { s_pitch, 0, 0, 0 };
    return sws_scale(s_sws, s_frame->data, s_frame->linesize, 0, s_height,
                     dst, stride) == s_height;
}

static BOOL DecodeNext(void)
{
    if (s_eof) return FALSE;
    for (;;) {
        int ret = avcodec_receive_frame(s_vctx, s_frame);
        if (ret == 0) {
            AVStream* st = s_fmt->streams[s_vstream];
            int64_t pts = s_frame->best_effort_timestamp;
            if (pts == AV_NOPTS_VALUE) pts = s_frame->pts;
            if (pts != AV_NOPTS_VALUE && st->time_base.den > 0) {
                if (s_tsOrigin == AV_NOPTS_VALUE) s_tsOrigin = pts;
                const double sec = (double)(pts - s_tsOrigin) *
                    av_q2d(st->time_base);
                s_frameIndex = (int)(sec * s_fps + 0.5);
            } else {
                ++s_frameIndex;
            }
            BOOL converted = ConvertFrame();
            av_frame_unref(s_frame);
            return converted;
        }
        if (ret == AVERROR_EOF) { s_eof = TRUE; return FALSE; }
        if (ret != AVERROR(EAGAIN)) { s_eof = TRUE; return FALSE; }
        if (s_flushed) { s_eof = TRUE; return FALSE; }

        ret = av_read_frame(s_fmt, s_pkt);
        while (ret >= 0 && s_pkt->stream_index != s_vstream) {
            av_packet_unref(s_pkt);
            ret = av_read_frame(s_fmt, s_pkt);
        }
        if (ret < 0) {
            s_flushed = TRUE;
            avcodec_send_packet(s_vctx, NULL);
            continue;
        }
        ret = avcodec_send_packet(s_vctx, s_pkt);
        av_packet_unref(s_pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            s_eof = TRUE;
            return FALSE;
        }
    }
}

static void Present(void)
{
    if (!s_rgba || s_width <= 0 || s_height <= 0) return;
    MarniDX* dx = Marni_DX();
    if (!dx) return;
    if (s_tex == MARNI_NULL_HANDLE || s_texW != s_width || s_texH != s_height) {
        if (s_tex != MARNI_NULL_HANDLE) dx->DestroyTexture(s_tex);
        int ow = 0, oh = 0;
        s_tex = dx->CreateTexture(s_width, s_height, 32, s_rgba, &ow, &oh);
        s_texW = s_width;
        s_texH = s_height;
        if (s_tex == MARNI_NULL_HANDLE) return;
    } else {
        dx->UpdateTexturePixels(s_tex, s_rgba, s_width, s_height, 32);
    }
    DWORD bw = 0, bh = 0;
    dx->GetBackBufferSize(&bw, &bh);
    if (!bw || !bh) return;
    dx->Clear(0, 0, 0, 1);
    int ms = s_frameIndex > 0 ? (int)(s_frameIndex * 1000.0 / s_fps + 0.5) : 0;
    if (!s_overlayCallback || !s_overlayCallback(s_tex, ms)) {
        dx->DrawSprite(0, 0, (float)bw, (float)bh, 0, 0, 1, 1,
                       0xFFFFFFFFu, s_tex, MARNI_SAMPLER_POINT,
                       MARNI_BLEND_DISABLE);
    }
    dx->Present();
    if (!s_loggedFirstPresent) {
        dbg_printf("[VIDEO] FFmpeg first frame presented: frame=%d texture=%dx%d\n",
                   s_frameIndex, s_width, s_height);
        s_loggedFirstPresent = TRUE;
    }
}

static void StartAudioSegment(void)
{
    if (!s_useAudio || !s_wave) return;
    if (s_hdrPrepared) {
        waveOutReset(s_wave);
        waveOutUnprepareHeader(s_wave, &s_hdr, sizeof(s_hdr));
        s_hdrPrepared = FALSE;
    }
    ZeroMemory(&s_hdr, sizeof(s_hdr));
    long start = s_audioStart < 0 ? 0 : s_audioStart;
    long end = s_audioEnd > start ? s_audioEnd : s_audioFrames;
    if (end > s_audioFrames) end = s_audioFrames;
    if (end <= start) return;
    s_hdr.lpData = (LPSTR)(s_audio + (size_t)start * 2);
    s_hdr.dwBufferLength = (DWORD)((end - start) * 2 * sizeof(short));
    if (waveOutPrepareHeader(s_wave, &s_hdr, sizeof(s_hdr)) != MMSYSERR_NOERROR)
        return;
    s_hdrPrepared = TRUE;
    if (waveOutWrite(s_wave, &s_hdr, sizeof(s_hdr)) != MMSYSERR_NOERROR) {
        waveOutUnprepareHeader(s_wave, &s_hdr, sizeof(s_hdr));
        s_hdrPrepared = FALSE;
    }
}

static BOOL SeekTo(int frame)
{
    if (frame < 0) frame = 0;
    AVStream* st = s_fmt->streams[s_vstream];
    const AVRational frameRate = st->avg_frame_rate.num > 0
        ? st->avg_frame_rate : st->r_frame_rate;
    int64_t first = st->start_time == AV_NOPTS_VALUE ? 0 : st->start_time;
    if (s_tsOrigin != AV_NOPTS_VALUE) first = s_tsOrigin;
    int64_t ts = first + av_rescale_q(frame,
        av_inv_q(frameRate.num > 0 ? frameRate : (AVRational){30, 1}),
        st->time_base);
    av_seek_frame(s_fmt, s_vstream, ts, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(s_vctx);
    s_eof = s_flushed = FALSE;
    s_frameIndex = -1;
    while (s_frameIndex < frame && !s_eof) {
        if (!DecodeNext()) break;
    }
    if (s_useAudio) {
        s_audioStart = (long)((double)frame / s_fps * s_audioRate);
        s_audioEnd = s_playToFrame > 0
            ? (long)((double)s_playToFrame / s_fps * s_audioRate)
            : s_audioFrames;
        StartAudioSegment();
    }
    s_baseFrame = frame;
    s_startTicks = plat_time_ms();
    return s_rgba != NULL;
}

static BOOL Open(const char* path, int playToMs)
{
    Release();
    if (avformat_open_input(&s_fmt, path, NULL, NULL) < 0 ||
        avformat_find_stream_info(s_fmt, NULL) < 0) {
        plat_debug_output("[VIDEO] FFmpeg could not parse MP4\n");
        Release();
        return FALSE;
    }
    s_vstream = av_find_best_stream(s_fmt, AVMEDIA_TYPE_VIDEO, -1, -1,
                                    NULL, 0);
    s_astream = av_find_best_stream(s_fmt, AVMEDIA_TYPE_AUDIO, -1, -1,
                                    NULL, 0);
    if (s_vstream < 0) { Release(); return FALSE; }

    AVStream* vst = s_fmt->streams[s_vstream];
    const AVCodec* vc = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!vc) { Release(); return FALSE; }
    s_vctx = avcodec_alloc_context3(vc);
    if (!s_vctx || avcodec_parameters_to_context(s_vctx, vst->codecpar) < 0 ||
        avcodec_open2(s_vctx, vc, NULL) < 0) { Release(); return FALSE; }
    s_width = s_vctx->width;
    s_height = s_vctx->height;
    AVRational rate = vst->avg_frame_rate.num > 0
        ? vst->avg_frame_rate : vst->r_frame_rate;
    s_fps = rate.num > 0 && rate.den > 0 ? av_q2d(rate) : 30.0;
    if (s_fps < 1.0 || s_fps > 120.0) s_fps = 30.0;

    if (s_astream >= 0) {
        AVStream* ast = s_fmt->streams[s_astream];
        const AVCodec* ac = avcodec_find_decoder(ast->codecpar->codec_id);
        if (ac) {
            s_actx = avcodec_alloc_context3(ac);
            if (!s_actx || avcodec_parameters_to_context(s_actx,
                    ast->codecpar) < 0 || avcodec_open2(s_actx, ac, NULL) < 0) {
                if (s_actx) avcodec_free_context(&s_actx);
                s_astream = -1;
            }
        } else {
            s_astream = -1;
        }
    }
    s_pkt = av_packet_alloc();
    s_frame = av_frame_alloc();
    if (!s_pkt || !s_frame) { Release(); return FALSE; }

    const BOOL audioReady = s_actx ? DecodeAudio() : FALSE;
    s_playToFrame = playToMs > 0
        ? (int)(playToMs * s_fps / 1000.0 + 0.5) : 0;
    s_active = TRUE;
    s_endEvent = FALSE;
    if (!SeekTo(0)) { Release(); return FALSE; }
    Present();
    dbg_printf("[VIDEO] FFmpeg MP4 opened: %s (%dx%d, %.3f fps, video=%s, audio=%s)\n",
        path, s_width, s_height, s_fps,
        avcodec_get_name(vc->id),
        audioReady && s_useAudio ? avcodec_get_name(s_actx->codec_id) : "silent");
    return TRUE;
}

static void PlayFrom(int fromMs)
{
    if (!s_fmt || !s_vctx) return;
    s_playToFrame = 0;
    s_endEvent = FALSE;
    s_eof = FALSE;
    s_active = TRUE;
    SeekTo((int)(fromMs * s_fps / 1000.0 + 0.5));
    Present();
}

static void Stop(void) { s_active = FALSE; ReleaseAudio(); }
static void Close(void) { Release(); }
static BOOL IsActive(void) { return s_active; }
static BOOL HasMedia(void) { return s_fmt != NULL || s_active; }
static BOOL TakeEndEvent(void)
{
    if (!s_endEvent) return FALSE;
    s_endEvent = FALSE;
    return TRUE;
}

static void Tick(void)
{
    if (!s_active || !s_vctx) return;
    const double wall = (double)(plat_time_ms() - s_startTicks) / 1000.0;
    double elapsed = wall;
    if (s_useAudio && s_wave) {
        MMTIME pos = {};
        pos.wType = TIME_SAMPLES;
        if (waveOutGetPosition(s_wave, &pos, sizeof(pos)) == MMSYSERR_NOERROR &&
            pos.wType == TIME_SAMPLES) {
            const double audio = (double)((long)pos.u.sample - s_audioStart) /
                                 s_audioRate;
            if (audio >= wall - 0.5 && audio <= wall + 0.25) elapsed = audio;
        }
    }
    const int target = s_baseFrame + (int)(elapsed * s_fps);
    while (s_frameIndex < target && !s_eof) {
        if (!DecodeNext()) break;
    }
    Present();
    if ((s_playToFrame > 0 && s_frameIndex >= s_playToFrame) ||
        (s_eof && target > s_frameIndex)) {
        s_active = FALSE;
        s_endEvent = TRUE;
    }
}

} // namespace ffmpeg_win

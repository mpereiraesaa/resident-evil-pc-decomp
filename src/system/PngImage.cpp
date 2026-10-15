// PngImage.cpp - Minimal PNG reader for port-only assets. See PngImage.h.
//
// The inflate follows RFC 1951 directly (canonical Huffman decode in the style
// of zlib's puff.c); the PNG side follows the PNG specification, section 9
// (filters) and 11 (chunks). Chunk CRCs and the zlib Adler-32 are not checked:
// the files are local assets, and a damaged stream still fails the size checks.
#include "PngImage.h"

#include <string.h>

namespace {

// ---------------------------------------------------------------------------
// Inflate
// ---------------------------------------------------------------------------

const int kMaxBits = 15;

struct BitStream {
    const unsigned char* in;
    size_t inSize;
    size_t inPos;
    unsigned int bitBuf;
    int bitCount;
    bool error;
    std::vector<unsigned char>* out;
    size_t outLimit;
};

int GetBits(BitStream& s, int need)
{
    unsigned int val = s.bitBuf;
    while (s.bitCount < need) {
        if (s.inPos >= s.inSize) {
            s.error = true;
            return 0;
        }
        val |= static_cast<unsigned int>(s.in[s.inPos++]) << s.bitCount;
        s.bitCount += 8;
    }
    s.bitBuf = val >> need;
    s.bitCount -= need;
    return static_cast<int>(val & ((1u << need) - 1u));
}

struct Huffman {
    short count[kMaxBits + 1];  // codes of each length
    short symbol[288];          // symbols ordered by code
};

// Build a canonical decoding table from per-symbol code lengths. Incomplete
// codes are accepted (a distance code may legally have a single symbol);
// over-subscribed ones are not.
bool BuildHuffman(Huffman& h, const short* length, int n)
{
    memset(h.count, 0, sizeof(h.count));
    for (int i = 0; i < n; ++i) h.count[length[i]]++;
    if (h.count[0] == n) return true;  // no codes at all

    int left = 1;
    for (int len = 1; len <= kMaxBits; ++len) {
        left <<= 1;
        left -= h.count[len];
        if (left < 0) return false;
    }

    short offs[kMaxBits + 1];
    offs[1] = 0;
    for (int len = 1; len < kMaxBits; ++len) offs[len + 1] = offs[len] + h.count[len];
    for (int i = 0; i < n; ++i) {
        if (length[i] != 0) h.symbol[offs[length[i]]++] = static_cast<short>(i);
    }
    return true;
}

int Decode(BitStream& s, const Huffman& h)
{
    int code = 0;
    int first = 0;
    int index = 0;
    for (int len = 1; len <= kMaxBits; ++len) {
        code |= GetBits(s, 1);
        if (s.error) return -1;
        const int count = h.count[len];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    s.error = true;
    return -1;
}

bool PutByte(BitStream& s, unsigned char b)
{
    if (s.out->size() >= s.outLimit) return false;
    s.out->push_back(b);
    return true;
}

bool InflateStored(BitStream& s)
{
    // A stored block starts on a byte boundary.
    s.bitBuf = 0;
    s.bitCount = 0;
    if (s.inPos + 4 > s.inSize) return false;
    const unsigned int len = s.in[s.inPos] | (s.in[s.inPos + 1] << 8);
    const unsigned int nlen = s.in[s.inPos + 2] | (s.in[s.inPos + 3] << 8);
    s.inPos += 4;
    if (len != (~nlen & 0xFFFFu)) return false;
    if (s.inPos + len > s.inSize) return false;
    for (unsigned int i = 0; i < len; ++i) {
        if (!PutByte(s, s.in[s.inPos + i])) return false;
    }
    s.inPos += len;
    return true;
}

const short kLenBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
const short kLenExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
const short kDistBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577 };
const short kDistExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };

bool InflateCodes(BitStream& s, const Huffman& lencode, const Huffman& distcode)
{
    for (;;) {
        int symbol = Decode(s, lencode);
        if (symbol < 0) return false;
        if (symbol < 256) {
            if (!PutByte(s, static_cast<unsigned char>(symbol))) return false;
            continue;
        }
        if (symbol == 256) return true;

        symbol -= 257;
        if (symbol >= 29) return false;
        const int len = kLenBase[symbol] + GetBits(s, kLenExtra[symbol]);
        symbol = Decode(s, distcode);
        if (symbol < 0 || symbol >= 30) return false;
        const size_t dist = static_cast<size_t>(kDistBase[symbol]) +
                            GetBits(s, kDistExtra[symbol]);
        if (s.error || dist > s.out->size()) return false;
        for (int i = 0; i < len; ++i) {
            if (!PutByte(s, (*s.out)[s.out->size() - dist])) return false;
        }
    }
}

bool InflateFixed(BitStream& s)
{
    static Huffman lencode;
    static Huffman distcode;
    static bool built = false;
    if (!built) {
        short lengths[288];
        int i = 0;
        for (; i < 144; ++i) lengths[i] = 8;
        for (; i < 256; ++i) lengths[i] = 9;
        for (; i < 280; ++i) lengths[i] = 7;
        for (; i < 288; ++i) lengths[i] = 8;
        BuildHuffman(lencode, lengths, 288);
        for (i = 0; i < 30; ++i) lengths[i] = 5;
        BuildHuffman(distcode, lengths, 30);
        built = true;
    }
    return InflateCodes(s, lencode, distcode);
}

bool InflateDynamic(BitStream& s)
{
    static const short kOrder[19] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

    const int nlen = GetBits(s, 5) + 257;
    const int ndist = GetBits(s, 5) + 1;
    const int ncode = GetBits(s, 4) + 4;
    if (s.error || nlen > 286 || ndist > 30) return false;

    short lengths[320];
    int index = 0;
    for (; index < ncode; ++index) lengths[kOrder[index]] = static_cast<short>(GetBits(s, 3));
    for (; index < 19; ++index) lengths[kOrder[index]] = 0;
    if (s.error) return false;

    Huffman lencode;
    Huffman distcode;
    if (!BuildHuffman(lencode, lengths, 19)) return false;

    index = 0;
    while (index < nlen + ndist) {
        int symbol = Decode(s, lencode);
        if (symbol < 0) return false;
        if (symbol < 16) {
            lengths[index++] = static_cast<short>(symbol);
            continue;
        }
        short len = 0;
        int repeat;
        if (symbol == 16) {
            if (index == 0) return false;
            len = lengths[index - 1];
            repeat = 3 + GetBits(s, 2);
        } else if (symbol == 17) {
            repeat = 3 + GetBits(s, 3);
        } else {
            repeat = 11 + GetBits(s, 7);
        }
        if (s.error || index + repeat > nlen + ndist) return false;
        while (repeat-- > 0) lengths[index++] = len;
    }
    if (lengths[256] == 0) return false;  // no end-of-block code

    if (!BuildHuffman(lencode, lengths, nlen)) return false;
    if (!BuildHuffman(distcode, lengths + nlen, ndist)) return false;
    return InflateCodes(s, lencode, distcode);
}

// Inflate a zlib stream (RFC 1950 header + deflate data) into `out`, which
// must not grow past `outLimit` bytes.
bool ZlibInflate(const unsigned char* in, size_t inSize,
                 std::vector<unsigned char>* out, size_t outLimit)
{
    if (inSize < 2) return false;
    const unsigned int cmf = in[0];
    const unsigned int flg = in[1];
    if ((cmf & 0x0F) != 8 || ((cmf << 8) | flg) % 31 != 0) return false;
    if (flg & 0x20) return false;  // preset dictionary

    BitStream s;
    s.in = in;
    s.inSize = inSize;
    s.inPos = 2;
    s.bitBuf = 0;
    s.bitCount = 0;
    s.error = false;
    s.out = out;
    s.outLimit = outLimit;

    int last;
    do {
        last = GetBits(s, 1);
        const int type = GetBits(s, 2);
        if (s.error) return false;
        bool ok;
        if (type == 0) ok = InflateStored(s);
        else if (type == 1) ok = InflateFixed(s);
        else if (type == 2) ok = InflateDynamic(s);
        else ok = false;
        if (!ok || s.error) return false;
    } while (!last);
    return true;
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------

unsigned int ReadBE32(const unsigned char* p)
{
    return (static_cast<unsigned int>(p[0]) << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
}

int Paeth(int a, int b, int c)
{
    const int p = a + b - c;
    const int pa = p > a ? p - a : a - p;
    const int pb = p > b ? p - b : b - p;
    const int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

}  // namespace

static bool PngDecodeInternal(const unsigned char* data, size_t size,
                              std::vector<unsigned char>* firstChannel,
                              std::vector<unsigned char>* rgbaOut,
                              int* width, int* height)
{
    static const unsigned char kSig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    if (data == NULL || (firstChannel == NULL && rgbaOut == NULL) ||
        size < 8 || memcmp(data, kSig, 8) != 0) {
        return false;
    }

    unsigned int w = 0;
    unsigned int h = 0;
    int depth = 0;
    int colorType = -1;
    unsigned char palette[256 * 3];
    unsigned char paletteAlpha[256];
    memset(paletteAlpha, 255, sizeof(paletteAlpha));
    int paletteCount = 0;
    unsigned char transparency[6] = {};
    size_t transparencySize = 0;
    std::vector<unsigned char> idat;

    size_t pos = 8;
    bool sawEnd = false;
    while (!sawEnd && pos + 12 <= size) {
        const unsigned int len = ReadBE32(data + pos);
        const unsigned char* type = data + pos + 4;
        const unsigned char* body = data + pos + 8;
        if (len > size - pos - 12) return false;

        if (memcmp(type, "IHDR", 4) == 0) {
            if (len < 13) return false;
            w = ReadBE32(body);
            h = ReadBE32(body + 4);
            depth = body[8];
            colorType = body[9];
            if (body[10] != 0 || body[11] != 0) return false;  // compression / filter method
            if (body[12] != 0) return false;                   // Adam7 not supported
        } else if (memcmp(type, "PLTE", 4) == 0) {
            if (len % 3 != 0 || len > sizeof(palette)) return false;
            memcpy(palette, body, len);
            paletteCount = static_cast<int>(len / 3);
        } else if (memcmp(type, "tRNS", 4) == 0) {
            if (len > sizeof(transparency)) return false;
            memcpy(transparency, body, len);
            transparencySize = len;
        } else if (memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), body, body + len);
        } else if (memcmp(type, "IEND", 4) == 0) {
            sawEnd = true;
        }
        pos += 12 + len;
    }
    if (w == 0 || h == 0 || w > 16384 || h > 16384 || idat.empty()) return false;
    if (colorType == 3 && transparencySize > static_cast<size_t>(paletteCount)) return false;
    if (colorType == 0 && transparencySize != 0 && transparencySize != 2) return false;
    if (colorType == 2 && transparencySize != 0 && transparencySize != 6) return false;
    if (colorType != 0 && colorType != 2 && colorType != 3 && transparencySize != 0) return false;
    if (colorType == 3 && transparencySize != 0) {
        memcpy(paletteAlpha, transparency, transparencySize);
    }

    int channels;
    switch (colorType) {
    case 0: channels = 1; break;
    case 2: channels = 3; break;
    case 3: channels = 1; break;
    case 4: channels = 2; break;
    case 6: channels = 4; break;
    default: return false;
    }
    const bool depthOk =
        (colorType == 0 && (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)) ||
        (colorType == 3 && (depth == 1 || depth == 2 || depth == 4 || depth == 8)) ||
        ((colorType == 2 || colorType == 4 || colorType == 6) && (depth == 8 || depth == 16));
    if (!depthOk) return false;
    if (colorType == 3 && paletteCount == 0) return false;

    const size_t rowBytes = (static_cast<size_t>(w) * channels * depth + 7) / 8;
    const size_t pixelBytes = (channels * depth + 7) / 8;  // filter stride, >= 1
    const size_t rawSize = (rowBytes + 1) * h;
    if (rawSize > 256u * 1024u * 1024u) return false;

    std::vector<unsigned char> raw;
    raw.reserve(rawSize);
    if (!ZlibInflate(idat.data(), idat.size(), &raw, rawSize)) return false;
    if (raw.size() != rawSize) return false;

    // Undo the per-row filters in place; row y's data starts after its filter byte.
    for (unsigned int y = 0; y < h; ++y) {
        unsigned char* row = &raw[y * (rowBytes + 1)];
        const unsigned char filter = row[0];
        unsigned char* cur = row + 1;
        const unsigned char* prev = y > 0 ? cur - (rowBytes + 1) : NULL;
        for (size_t i = 0; i < rowBytes; ++i) {
            const int a = i >= pixelBytes ? cur[i - pixelBytes] : 0;
            const int b = prev != NULL ? prev[i] : 0;
            const int c = (prev != NULL && i >= pixelBytes) ? prev[i - pixelBytes] : 0;
            int v = cur[i];
            switch (filter) {
            case 0: break;
            case 1: v += a; break;
            case 2: v += b; break;
            case 3: v += (a + b) >> 1; break;
            case 4: v += Paeth(a, b, c); break;
            default: return false;
            }
            cur[i] = static_cast<unsigned char>(v);
        }
    }

    const size_t pixelCount = static_cast<size_t>(w) * h;
    // Bound output allocation as well as inflated scanlines: indexed PNGs can
    // expand from one byte per pixel to four bytes per pixel here.
    if (pixelCount > 16u * 1024u * 1024u) return false;
    if (firstChannel != NULL) firstChannel->assign(pixelCount, 0);
    if (rgbaOut != NULL) rgbaOut->assign(pixelCount * 4, 0);
    const int sampleBytes = depth == 16 ? 2 : 1;
    for (unsigned int y = 0; y < h; ++y) {
        const unsigned char* row = &raw[y * (rowBytes + 1) + 1];
        for (unsigned int x = 0; x < w; ++x) {
            unsigned int sample = 0;
            unsigned char red = 0, green = 0, blue = 0, alpha = 255;
            if (depth < 8) {
                const unsigned int bit = x * depth;
                const int mask = (1 << depth) - 1;
                sample = (row[bit >> 3] >> (8 - depth - (bit & 7))) & mask;
                if (colorType == 3) {
                    if (sample < static_cast<unsigned int>(paletteCount)) {
                        red = palette[sample * 3];
                        green = palette[sample * 3 + 1];
                        blue = palette[sample * 3 + 2];
                        alpha = paletteAlpha[sample];
                    }
                } else {
                    red = green = blue = static_cast<unsigned char>(sample * 255 / mask);
                    if (transparencySize == 2 && transparency[0] == 0 &&
                        transparency[1] == sample) alpha = 0;
                }
            } else {
                const unsigned char* px = row + static_cast<size_t>(x) * channels * sampleBytes;
                const unsigned int s0 = sampleBytes == 2
                    ? (static_cast<unsigned int>(px[0]) << 8) | px[1] : px[0];
                const unsigned int s1 = channels > 1 ? (sampleBytes == 2
                    ? (static_cast<unsigned int>(px[sampleBytes]) << 8) | px[sampleBytes + 1]
                    : px[sampleBytes]) : 0;
                const unsigned int s2 = channels > 2 ? (sampleBytes == 2
                    ? (static_cast<unsigned int>(px[2 * sampleBytes]) << 8) | px[2 * sampleBytes + 1]
                    : px[2 * sampleBytes]) : 0;
                const unsigned char c0 = px[0];
                if (colorType == 0) {
                    red = green = blue = c0;
                    if (transparencySize == 2 &&
                        ((static_cast<unsigned int>(transparency[0]) << 8) |
                         transparency[1]) == s0) alpha = 0;
                } else if (colorType == 2) {
                    red = c0; green = px[sampleBytes]; blue = px[2 * sampleBytes];
                    if (transparencySize == 6 &&
                        ((static_cast<unsigned int>(transparency[0]) << 8) | transparency[1]) == s0 &&
                        ((static_cast<unsigned int>(transparency[2]) << 8) | transparency[3]) == s1 &&
                        ((static_cast<unsigned int>(transparency[4]) << 8) | transparency[5]) == s2) alpha = 0;
                } else if (colorType == 3) {
                    sample = c0;
                    if (sample < static_cast<unsigned int>(paletteCount)) {
                        red = palette[sample * 3];
                        green = palette[sample * 3 + 1];
                        blue = palette[sample * 3 + 2];
                        alpha = paletteAlpha[sample];
                    }
                } else if (colorType == 4) {
                    red = green = blue = c0;
                    alpha = px[sampleBytes];
                } else { // RGBA
                    red = c0; green = px[sampleBytes]; blue = px[2 * sampleBytes];
                    alpha = px[3 * sampleBytes];
                }
            }

            const size_t index = static_cast<size_t>(y) * w + x;
            if (firstChannel != NULL) {
                (*firstChannel)[index] = alpha == 0 ? 0 : red;
            }
            if (rgbaOut != NULL) {
                unsigned char* dst = &(*rgbaOut)[index * 4];
                dst[0] = red; dst[1] = green; dst[2] = blue; dst[3] = alpha;
            }
        }
    }

    if (width != NULL) *width = static_cast<int>(w);
    if (height != NULL) *height = static_cast<int>(h);
    return true;
}

bool PngDecodeFirstChannel(const unsigned char* data, size_t size,
                           std::vector<unsigned char>* out,
                           int* width, int* height)
{
    if (out == NULL) return false;
    return PngDecodeInternal(data, size, out, NULL, width, height);
}

bool PngDecodeRGBA(const unsigned char* data, size_t size,
                   std::vector<unsigned char>* out,
                   int* width, int* height)
{
    if (out == NULL) return false;
    return PngDecodeInternal(data, size, NULL, out, width, height);
}

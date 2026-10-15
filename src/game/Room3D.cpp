// Room3D.cpp - opt-in static room blockout renderer.
//
// The first slice deliberately accepts a small OBJ/MTL subset: room-space
// positions, UVs, diffuse PNGs, and polygon faces. Geometry is supplied by an
// asset mod at Stage1/room1180.obj; the original RDT, scripts, camera cuts,
// collision, and PAKs remain authoritative.
#include "../Globals.h"
#include "../DebugPrint.h"
#include "../platform/platform.h"
#include "../system/AssetPath.h"
#include "../system/PngImage.h"
#include "../marni/MarniDX.h"
#include "../marni/MarniSystem.h"
#include "SpriteRenderer.h"
#include "TmdRenderer.h"
#include "dc/ArrangeStages.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

bool ModernCamera_GetView(float from[3], float to[3]);   // ModernCamera.cpp
bool ModernCamera_Active(void);                         // ModernCamera.cpp
float ModernCamera_WallClipDepth(void);                 // ModernCamera.cpp

namespace {

const size_t kMaxObjBytes = 16 * 1024 * 1024;
const size_t kMaxMtlBytes = 1024 * 1024;
const size_t kMaxPngBytes = 16 * 1024 * 1024;
const size_t kMaxTexturePixels = 4096 * 4096;
const size_t kMaxVertices = 400000;
const size_t kMaxTexcoords = 400000;
const size_t kMaxTriangles = 200000;
const size_t kMaxMaterials = 128;
const size_t kMaxMaterialName = 64;
const size_t kMaxMaterialPath = 256;
const int kTrianglesPerBatch = 1024;
const float kNearFloor = 1.0f;

struct Vertex {
    float x, y, z;
    float u, v;
    float r, g, b;
};

struct Triangle {
    Vertex v[3];
    int material;
    unsigned int hiddenCameras;
    bool shell;   // wall/floor/ceiling/door/gallery/stair geometry (camera collision)
    signed char projection;   // original camera whose background paints it, or -1
};

struct CameraVisibility {
    std::string group;
    unsigned int hiddenCameras;
};

struct Material {
    char name[kMaxMaterialName];
    char diffuseMap[kMaxMaterialPath];
    float r, g, b;
    MarniHandle texture;
    bool hasDiffuseColor;
    bool used;
};

struct TextToken {
    const char* data;
    size_t size;
};

struct RoomCameraBasis {
    float from[3];
    float right[3];
    float down[3];
    float forward[3];
};

std::vector<Triangle> s_triangles;
std::vector<Material> s_materials;
bool s_loadAttempted = false;
bool s_loaded = false;
int s_loadedFileStage = -1;
int s_loadedRoomId = -1;
int s_loadedVariant = -1;
// Cameras whose original pre-rendered background is kept (cutscene and
// aerial views placed outside the room); set by "backdrop N" lines.
unsigned int s_backdropCameras = 0;
// Per-camera brightness for rooms whose originals have a lights-off camera
// set over the same geometry ("shade N F" lines); 1 leaves the bake alone.
float s_cameraShade[32];
unsigned int s_shadedCameras = 0;
float s_activeShade = 1.0f;
// Original backgrounds projected onto the mesh (roomSRRV.proj sidecar, see
// tools/room3d_projection.py). Loaded through the normal asset resolution, so
// an HD pack's Stage1/RCxxxx.webp replaces the 320x240 PAK automatically.
MarniHandle s_projectionTextures[32];
bool s_hasProjection = false;

void MaterialColor(const char* name, float* r, float* g, float* b)
{
    // Flat, restrained fallback colors keep malformed/missing PNG materials
    // useful as blockout geometry instead of disabling the room mesh.
    if (strcmp(name, "floor") == 0)      { *r = 0.18f; *g = 0.20f; *b = 0.24f; }
    else if (strcmp(name, "wall") == 0)  { *r = 0.20f; *g = 0.23f; *b = 0.28f; }
    else if (strcmp(name, "ceiling") == 0) { *r = 0.12f; *g = 0.13f; *b = 0.16f; }
    else if (strcmp(name, "door") == 0)  { *r = 0.13f; *g = 0.18f; *b = 0.25f; }
    else if (strcmp(name, "wood") == 0)  { *r = 0.34f; *g = 0.27f; *b = 0.19f; }
    else if (strcmp(name, "metal") == 0) { *r = 0.30f; *g = 0.34f; *b = 0.36f; }
    else                       { *r = 0.28f; *g = 0.30f; *b = 0.33f; }
}

bool IsTextSpace(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

bool NextLine(const std::string& source, size_t* offset, TextToken* line)
{
    if (offset == NULL || line == NULL || *offset > source.size()) return false;
    const size_t start = *offset;
    size_t end = source.find('\n', start);
    if (end == std::string::npos) end = source.size();
    *offset = end == source.size() ? source.size() + 1 : end + 1;
    if (end > start && source[end - 1] == '\r') --end;
    line->data = source.c_str() + start;
    line->size = end - start;
    return true;
}

bool NextToken(const TextToken& line, size_t* offset, TextToken* token)
{
    if (offset == NULL || token == NULL || *offset > line.size) return false;
    while (*offset < line.size && IsTextSpace(line.data[*offset])) ++*offset;
    if (*offset >= line.size || line.data[*offset] == '#') return false;
    const size_t start = *offset;
    while (*offset < line.size && !IsTextSpace(line.data[*offset]) &&
           line.data[*offset] != '#') ++*offset;
    token->data = line.data + start;
    token->size = *offset - start;
    return token->size != 0;
}

bool TokenEquals(const TextToken& token, const char* value)
{
    const size_t length = strlen(value);
    return token.size == length && memcmp(token.data, value, length) == 0;
}

std::string TokenString(const TextToken& token)
{
    return std::string(token.data, token.size);
}

bool ParseFloatToken(const TextToken& token, float* out)
{
    if (out == NULL || token.size == 0 || token.size >= 64) return false;
    char buffer[64];
    memcpy(buffer, token.data, token.size);
    buffer[token.size] = '\0';
    char* end = NULL;
    const float value = strtof(buffer, &end);
    if (end != buffer + token.size || !std::isfinite(value)) return false;
    *out = value;
    return true;
}

// OBJ's common RGB extension follows XYZ. A lone homogeneous weight remains
// accepted for ordinary OBJ files, without changing the room-space positions.
bool ParseVertexColor(const TextToken& line, size_t offset, Vertex* vertex)
{
    float values[3] = {};
    size_t count = 0;
    TextToken token;
    vertex->r = vertex->g = vertex->b = 1.0f;
    while (NextToken(line, &offset, &token)) {
        if (count == 3 || !ParseFloatToken(token, &values[count])) return false;
        ++count;
    }
    if (count == 0 || count == 1) return true;
    if (count != 3) return false;
    for (size_t i = 0; i < 3; ++i)
        if (values[i] < 0.0f || values[i] > 1.0f) return false;
    vertex->r = values[0];
    vertex->g = values[1];
    vertex->b = values[2];
    return true;
}

bool ResolveObjIndex(const char* value, size_t length, size_t begin, size_t spanEnd,
                     size_t count, size_t* out)
{
    if (value == NULL || begin >= spanEnd || spanEnd > length || out == NULL)
        return false;
    char* parsedEnd = NULL;
    const char* start = value + begin;
    const long index = strtol(start, &parsedEnd, 10);
    if (parsedEnd == start || parsedEnd != value + spanEnd || index == 0)
        return false;

    const long resolved = index > 0 ? index - 1 : (long)count + index;
    if (resolved < 0 || (size_t)resolved >= count) return false;
    *out = (size_t)resolved;
    return true;
}

bool ParseFaceIndex(const TextToken& token, size_t vertexCount,
                    size_t texcoordCount, size_t* vertex, size_t* texcoord)
{
    size_t slash = token.size;
    for (size_t i = 0; i < token.size; ++i) {
        if (token.data[i] == '/') { slash = i; break; }
    }
    if (texcoord != NULL) *texcoord = static_cast<size_t>(-1);
    if (slash == token.size)
        return ResolveObjIndex(token.data, token.size, 0, token.size,
                               vertexCount, vertex);
    if (!ResolveObjIndex(token.data, token.size, 0, slash, vertexCount, vertex))
        return false;

    size_t secondSlash = token.size;
    for (size_t i = slash + 1; i < token.size; ++i) {
        if (token.data[i] == '/') { secondSlash = i; break; }
    }
    const size_t uvBegin = slash + 1;
    const size_t uvEnd = secondSlash;
    if (uvBegin == uvEnd) return true; // v//vn is valid and has no UV.
    return ResolveObjIndex(token.data, token.size, uvBegin, uvEnd,
                           texcoordCount, texcoord);
}

bool IsSafeRelativeAssetPath(const std::string& path)
{
    if (path.empty() || path.size() >= 512 || path[0] == '/' || path[0] == '\\' ||
        path.find(':') != std::string::npos) return false;
    size_t start = 0;
    while (start <= path.size()) {
        const size_t end = path.find_first_of("/\\", start);
        const size_t componentEnd = end == std::string::npos ? path.size() : end;
        if (componentEnd - start == 2 && path[start] == '.' && path[start + 1] == '.')
            return false;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return true;
}

bool ReadTextFile(const std::string& path, size_t maxBytes, std::string* out)
{
    if (path.size() >= 512) return false;
    char normalized[520];
    const char* filePath = plat_normalize_path(path.c_str(), normalized, sizeof(normalized));
    FILE* file = fopen(filePath, "rb");
    if (file == NULL) return false;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return false; }
    const long length = ftell(file);
    if (length <= 0 || (size_t)length > maxBytes || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }
    out->assign((size_t)length, '\0');
    const size_t read = fread(&(*out)[0], 1, out->size(), file);
    fclose(file);
    return read == out->size();
}

void LoadCameraVisibility(const std::string& objPath,
                          std::vector<CameraVisibility>* groups,
                          unsigned int* backdropCameras)
{
    const std::string path = objPath.substr(0, objPath.size() - 4) + ".cameras";
    std::string source;
    if (!ReadTextFile(path, 4096, &source)) return;
    size_t offset = 0;
    TextToken line;
    while (NextLine(source, &offset, &line)) {
        size_t tokenOffset = 0;
        TextToken command, indexToken, groupToken, extra;
        if (!NextToken(line, &tokenOffset, &command)) continue;
        float index = 0.0f;
        TextToken factorToken;
        float factor = 1.0f;
        if (TokenEquals(command, "shade") &&
            NextToken(line, &tokenOffset, &indexToken) &&
            NextToken(line, &tokenOffset, &factorToken) &&
            !NextToken(line, &tokenOffset, &extra) &&
            ParseFloatToken(indexToken, &index) &&
            ParseFloatToken(factorToken, &factor) &&
            index >= 0.0f && index < 32.0f && index == floorf(index) &&
            factor >= 0.0f && factor <= 4.0f) {
            s_cameraShade[(unsigned int)index] = factor;
            s_shadedCameras |= 1u << (unsigned int)index;
            continue;
        }
        if (TokenEquals(command, "backdrop") &&
            NextToken(line, &tokenOffset, &indexToken) &&
            !NextToken(line, &tokenOffset, &extra) &&
            ParseFloatToken(indexToken, &index) &&
            index >= 0.0f && index < 32.0f && index == floorf(index)) {
            *backdropCameras |= 1u << (unsigned int)index;
            continue;
        }
        if (!TokenEquals(command, "hide") ||
            !NextToken(line, &tokenOffset, &indexToken) ||
            !NextToken(line, &tokenOffset, &groupToken) ||
            NextToken(line, &tokenOffset, &extra) ||
            !ParseFloatToken(indexToken, &index) ||
            index < 0.0f || index >= 32.0f || index != floorf(index) ||
            groupToken.size >= 64) {
            dbg_printf("[room3d] ignoring invalid camera visibility in %s\n", path.c_str());
            continue;
        }
        CameraVisibility entry;
        entry.group = TokenString(groupToken);
        entry.hiddenCameras = 1u << (unsigned int)index;
        groups->push_back(entry);
    }
}

void LoadMaterialLibrary(const std::string& mtlPath, std::vector<Material>* materials)
{
    std::string source;
    if (!ReadTextFile(mtlPath, kMaxMtlBytes, &source)) {
        dbg_printf("[room3d] cannot read material library %s\n", mtlPath.c_str());
        return;
    }

    int current = -1;
    size_t lineOffset = 0;
    TextToken line;
    while (NextLine(source, &lineOffset, &line)) {
        size_t tokenOffset = 0;
        TextToken kind;
        if (!NextToken(line, &tokenOffset, &kind)) continue;
        if (TokenEquals(kind, "newmtl")) {
            TextToken nameToken;
            if (!NextToken(line, &tokenOffset, &nameToken)) {
                current = -1;
                continue;
            }
            if (nameToken.size >= kMaxMaterialName) {
                current = -1;
                continue;
            }
            size_t found = 0;
            while (found < materials->size() &&
                   (strlen((*materials)[found].name) != nameToken.size ||
                    memcmp((*materials)[found].name, nameToken.data, nameToken.size) != 0))
                ++found;
            if (found < materials->size()) {
                current = (int)found;
                continue;
            }
            if (materials->size() >= kMaxMaterials) { current = -1; continue; }
            Material material = {};
            memcpy(material.name, nameToken.data, nameToken.size);
            material.name[nameToken.size] = '\0';
            material.r = material.g = material.b = 1.0f;
            material.texture = MARNI_NULL_HANDLE;
            current = (int)materials->size();
            materials->push_back(material);
        } else if (current >= 0 && TokenEquals(kind, "Kd")) {
            Material& material = (*materials)[current];
            float r, g, b;
            TextToken rToken, gToken, bToken;
            if (NextToken(line, &tokenOffset, &rToken) &&
                NextToken(line, &tokenOffset, &gToken) &&
                NextToken(line, &tokenOffset, &bToken) &&
                ParseFloatToken(rToken, &r) && ParseFloatToken(gToken, &g) &&
                ParseFloatToken(bToken, &b)) {
                material.r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
                material.g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
                material.b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
                material.hasDiffuseColor = true;
            }
        } else if (current >= 0 && TokenEquals(kind, "map_Kd")) {
            // This minimal loader accepts the last token as a relative PNG
            // filename; map_Kd options and filenames containing spaces are
            // outside the initial asset interchange subset.
            TextToken filenameToken = {};
            TextToken token;
            while (NextToken(line, &tokenOffset, &token)) {
                filenameToken = token;
            }
            if (filenameToken.data != NULL && filenameToken.size < kMaxMaterialPath) {
                const std::string filename = TokenString(filenameToken);
                if (IsSafeRelativeAssetPath(filename)) {
                    memcpy((*materials)[current].diffuseMap,
                           filenameToken.data, filenameToken.size);
                    (*materials)[current].diffuseMap[filenameToken.size] = '\0';
                }
            }
        }
    }
}

int GetMaterialIndex(const TextToken& nameToken, std::vector<Material>* materials)
{
    if (nameToken.size >= kMaxMaterialName) return -1;
    for (size_t i = 0; i < materials->size(); ++i) {
        const char* name = (*materials)[i].name;
        if (strlen(name) == nameToken.size &&
            memcmp(name, nameToken.data, nameToken.size) == 0)
            return (int)i;
    }
    if (materials->size() >= kMaxMaterials) return -1;
    Material material = {};
    memcpy(material.name, nameToken.data, nameToken.size);
    material.name[nameToken.size] = '\0';
    MaterialColor(material.name, &material.r, &material.g, &material.b);
    material.texture = MARNI_NULL_HANDLE;
    const int index = (int)materials->size();
    materials->push_back(material);
    return index;
}

void LoadMaterialTextures(const std::string& objDirectory,
                          std::vector<Material>* materials)
{
    MarniDX* dx = Marni_DX();
    if (dx == NULL) return;
    for (size_t i = 0; i < materials->size(); ++i) {
        Material& material = (*materials)[i];
        if (!material.used || material.diffuseMap[0] == '\0') continue;
        const std::string textureName(material.diffuseMap);
        const std::string relative = objDirectory + textureName;
        if (!IsSafeRelativeAssetPath(textureName) || relative.size() >= 512)
            continue;

        char normalized[520];
        const char* texturePath = plat_normalize_path(relative.c_str(), normalized,
                                                       sizeof(normalized));
        FILE* file = fopen(texturePath, "rb");
        if (file == NULL) {
            dbg_printf("[room3d] missing material texture %s\n", texturePath);
            continue;
        }
        if (fseek(file, 0, SEEK_END) != 0) { fclose(file); continue; }
        const long length = ftell(file);
        if (length <= 0 || (size_t)length > kMaxPngBytes ||
            fseek(file, 0, SEEK_SET) != 0) {
            fclose(file);
            dbg_printf("[room3d] invalid material texture size: %s\n", texturePath);
            continue;
        }
        std::vector<unsigned char> encoded((size_t)length);
        const size_t read = fread(encoded.data(), 1, encoded.size(), file);
        fclose(file);
        if (read != encoded.size()) continue;

        std::vector<unsigned char> rgba;
        int width = 0, height = 0;
        if (!PngDecodeRGBA(encoded.data(), encoded.size(), &rgba, &width, &height) ||
            width <= 0 || height <= 0 || (size_t)width * height > kMaxTexturePixels) {
            dbg_printf("[room3d] invalid PNG material texture: %s\n", texturePath);
            continue;
        }
        material.texture = dx->CreateTexture(width, height, 32, rgba.data(), NULL, NULL);
        if (material.texture == MARNI_NULL_HANDLE)
            dbg_printf("[room3d] could not upload material texture: %s\n", texturePath);
    }
}

bool BasisFromPoints(RoomCameraBasis* basis, const float from[3], const float to[3]);

// The basis of original RDT camera `index` (the fixed view its background
// was rendered from) and its focal length.
bool RdtCameraBasis(int index, RoomCameraBasis* basis, float* focal)
{
    if (g_RdtPointer == NULL || index < 0 ||
        (unsigned int)index >= (unsigned int)g_RdtPointer->cameras_count)
        return false;
    const RDT_Camera& camera = ((RDT_Camera*)((char*)g_RdtPointer + sizeof(RDT)))[index];
    if (camera.roll != 0 || camera.fov <= 0) return false;
    const float from[3] = { (float)camera.cam_from_x, (float)camera.cam_from_y,
                            (float)camera.cam_from_z };
    const float to[3] = { (float)camera.cam_to_x, (float)camera.cam_to_y,
                          (float)camera.cam_to_z };
    *focal = (float)camera.fov;
    return BasisFromPoints(basis, from, to);
}

struct MaterialTint {
    std::string material;
    float r, g, b;
};

// "proj1 <faces>" then one character per OBJ face: '0'-'9', 'a'-'v' name the
// camera, '-' keeps the procedural material. "tint <material> r g b" lines
// shift a procedural material toward the art its projected faces show, so
// the faces no camera saw match their neighbours.
bool LoadProjection(const std::string& objPath, std::string* faces, size_t* faceCount,
                    std::vector<MaterialTint>* tints)
{
    static const char* env = std::getenv("RE1_ROOM3D_PROJECTION");
    if (env != NULL && env[0] == '0') return false;
    std::string source;
    if (!ReadTextFile(objPath.substr(0, objPath.size() - 4) + ".proj",
                      kMaxTriangles + 16384, &source))
        return false;
    size_t offset = 0, tokenOffset = 0;
    TextToken header, magic, countToken, body;
    float count = 0.0f;
    if (!NextLine(source, &offset, &header) ||
        !NextToken(header, &tokenOffset, &magic) || !TokenEquals(magic, "proj1") ||
        !NextToken(header, &tokenOffset, &countToken) ||
        !ParseFloatToken(countToken, &count) || count < 1.0f ||
        !NextLine(source, &offset, &body))
        return false;
    tokenOffset = 0;
    TextToken text;
    if (!NextToken(body, &tokenOffset, &text) || (float)text.size != count) return false;
    faces->assign(text.data, text.size);
    *faceCount = text.size;
    TextToken line;
    while (NextLine(source, &offset, &line)) {
        tokenOffset = 0;
        TextToken command, name, r, g, b;
        MaterialTint tint;
        if (NextToken(line, &tokenOffset, &command) && TokenEquals(command, "tint") &&
            NextToken(line, &tokenOffset, &name) && name.size < kMaxMaterialName &&
            NextToken(line, &tokenOffset, &r) && ParseFloatToken(r, &tint.r) &&
            NextToken(line, &tokenOffset, &g) && ParseFloatToken(g, &tint.g) &&
            NextToken(line, &tokenOffset, &b) && ParseFloatToken(b, &tint.b) &&
            tint.r >= 0.0f && tint.r <= 4.0f && tint.g >= 0.0f && tint.g <= 4.0f &&
            tint.b >= 0.0f && tint.b <= 4.0f) {
            tint.material = TokenString(name);
            tints->push_back(tint);
        }
    }
    return true;
}

int ProjectionCamera(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'v') return c - 'a' + 10;
    return -1;
}

// One camera's background as a texture: the pack's .webp when one resolves,
// otherwise the original PAK's 16-bit TIM.
MarniHandle LoadBackgroundTexture(int index)
{
    char path[sizeof(g_bgPathTemplate)];
    memcpy(path, g_bgPathTemplate, sizeof(path));
    // As load_room_bg: the revisit stages (and arrange STAGED/E) reuse the
    // backgrounds of the stage they revisit.
    const int fileStage = room_file_stage();
    const int stageDigit = STAGE_DATA_ROW_OF(fileStage) > 4 ? fileStage - 5 + 1 : fileStage + 1;
    path[GAME_DATA_PATH_IDX(0x0f)] = g_hexCharTable[stageDigit & 0xf];
    path[GAME_DATA_PATH_IDX(0x0b)] = path[GAME_DATA_PATH_IDX(0x0f)];
    path[GAME_DATA_PATH_IDX(0x10)] = g_hexCharTable[(g_roomId >> 4) & 0xf];
    path[GAME_DATA_PATH_IDX(0x11)] = g_hexCharTable[g_roomId & 0xf];
    path[GAME_DATA_PATH_IDX(0x12)] = g_hexCharTable[index & 0xf];

    MarniDX* dx = Marni_DX();
    char webp[sizeof(path) + 8];
    strcpy(webp, path);
    char* extension = strrchr(webp, '.');
    if (extension != NULL) {
        strcpy(extension, ".webp");
        char resolved[520];
        const char* imagePath = ResolveAssetRoot(webp, resolved, sizeof(resolved));
        size_t encodedSize = 0;
        void* encoded = plat_file_read_all(imagePath, &encodedSize);
        if (encoded != NULL) {
            char libraryResolved[520];
            const char* libraryPath = ResolveAssetRoot(
                GAME_DATA_ROOT "libwebp.dll", libraryResolved, sizeof(libraryResolved));
            int width = 0, height = 0;
            void* rgba = plat_webp_decode_rgba(libraryPath, encoded, encodedSize,
                                               &width, &height);
            free(encoded);
            if (rgba != NULL && width > 0 && height > 0 &&
                (size_t)width * height <= kMaxTexturePixels) {
                MarniHandle texture = dx->CreateTexture(width, height, 32, rgba, NULL, NULL);
                plat_webp_free(rgba);
                if (texture != MARNI_NULL_HANDLE) {
                    dbg_printf("[room3d] camera %d background %s (%dx%d)\n",
                               index, imagePath, width, height);
                    return texture;
                }
            } else {
                plat_webp_free(rgba);
            }
        }
    }

    char resolved[520];
    const char* pakPath = ResolveAssetRoot(path, resolved, sizeof(resolved));
    size_t pakSize = 0;
    void* pak = plat_file_read_all(pakPath, &pakSize);
    if (pak == NULL) return MARNI_NULL_HANDLE;
    std::vector<unsigned char> tim(sizeof(g_TimImageBuffer));
    unpack_pakfile_(pak, tim.data());
    free(pak);
    const int width = tim[16] | (tim[17] << 8);
    const int height = tim[18] | (tim[19] << 8);
    if (width <= 0 || height <= 0 || 20 + (size_t)width * height * 2 > tim.size())
        return MARNI_NULL_HANDLE;
    std::vector<unsigned int> rgba((size_t)width * height);
    for (size_t i = 0; i < rgba.size(); ++i) {
        const unsigned int px = tim[20 + i * 2] | (tim[21 + i * 2] << 8);
        const unsigned int r = (px & 0x1F) * 255 / 31;
        const unsigned int g = ((px >> 5) & 0x1F) * 255 / 31;
        const unsigned int b = ((px >> 10) & 0x1F) * 255 / 31;
        rgba[i] = 0xFF000000u | (b << 16) | (g << 8) | r;
    }
    return dx->CreateTexture(width, height, 32, rgba.data(), NULL, NULL);
}

bool LoadRoomMesh()
{
    // Mesh assets follow the same file-stage and character-variant selectors
    // as RDTs. Reset the previous room when a door transition changes any key.
    const int fileStage = room_file_stage();
    const int roomId = g_roomId;
    const int variant = (g_main_state_flags & MSF_CHAR_VARIANT) != 0 ? 1 : 0;
    if (fileStage != s_loadedFileStage || roomId != s_loadedRoomId ||
        variant != s_loadedVariant) {
        MarniDX* previousDx = Marni_DX();
        if (previousDx != NULL) {
            for (size_t i = 0; i < s_materials.size(); ++i) {
                if (s_materials[i].texture != MARNI_NULL_HANDLE)
                    previousDx->DestroyTexture(s_materials[i].texture);
            }
        }
        for (int k = 0; k < 32; ++k) {
            if (previousDx != NULL && s_projectionTextures[k] != MARNI_NULL_HANDLE)
                previousDx->DestroyTexture(s_projectionTextures[k]);
            s_projectionTextures[k] = MARNI_NULL_HANDLE;
        }
        s_hasProjection = false;
        s_triangles.clear();
        s_materials.clear();
        s_loaded = false;
        s_loadAttempted = false;
        s_backdropCameras = 0;
        s_shadedCameras = 0;
        s_loadedFileStage = fileStage;
        s_loadedRoomId = roomId;
        s_loadedVariant = variant;
    }
    if (Marni_DX() == NULL) return false;
    if (s_loadAttempted) return s_loaded;
    s_loadAttempted = true;

    char relativePath[96];
    sprintf_s(relativePath, sizeof(relativePath), "Stage%X/room%X%02X%d.obj",
              fileStage + 1, fileStage + 1, roomId, variant);
    std::string requestedPath = GAME_DATA_ROOT;
    requestedPath += relativePath;
    char resolved[520];
    char normalized[520];
    const char* path = ResolveAssetRoot(
        requestedPath.c_str(), resolved, sizeof(resolved));
    path = plat_normalize_path(path, normalized, sizeof(normalized));

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        dbg_printf("[room3d] no mesh at %s; keeping original background\n", path);
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return false; }
    const long length = ftell(file);
    if (length <= 0 || (size_t)length > kMaxObjBytes ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        dbg_printf("[room3d] invalid OBJ size: %s\n", path);
        return false;
    }

    std::string source((size_t)length, '\0');
    const size_t read = fread(&source[0], 1, source.size(), file);
    fclose(file);
    if (read != source.size()) {
        dbg_printf("[room3d] short OBJ read: %s\n", path);
        return false;
    }

    std::vector<Vertex> vertices;
    std::vector<Vertex> texcoords;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<CameraVisibility> visibility;
    LoadCameraVisibility(path, &visibility, &s_backdropCameras);
    std::string projection;
    size_t projectionFaces = 0;
    std::vector<MaterialTint> tints;
    const bool hasProjection = LoadProjection(path, &projection, &projectionFaces, &tints);
    size_t faceIndex = 0;
    unsigned int hiddenCameras = 0;
    bool currentShell = true;   // geometry before any group line is the shell
    static const char kDefaultMaterial[] = "default";
    TextToken currentMaterial = { kDefaultMaterial, sizeof(kDefaultMaterial) - 1 };
    size_t pathSep = std::string(path).find_last_of("/\\");
    const std::string objDirectory = pathSep == std::string::npos
        ? std::string() : std::string(path, pathSep + 1);
    size_t lineNumber = 0;
    size_t lineOffset = 0;
    TextToken line;
    while (NextLine(source, &lineOffset, &line)) {
        ++lineNumber;
        size_t tokenOffset = 0;
        TextToken kind;
        if (!NextToken(line, &tokenOffset, &kind)) continue;

        if (TokenEquals(kind, "g")) {
            hiddenCameras = 0;
            currentShell = false;
            TextToken group;
            while (NextToken(line, &tokenOffset, &group)) {
                // Builder group names: walls, wall_N, floor_N, ceiling_N,
                // door_N, gallery, stairs, shaft... are the room's shell;
                // furniture is ungrouped or carries its own names.
                static const char* const kShell[] = {
                    "wall", "floor", "ceiling", "door", "gallery", "stair", "shaft", "solid" };
                for (size_t k = 0; k < sizeof(kShell) / sizeof(kShell[0]); ++k) {
                    const size_t n = strlen(kShell[k]);
                    if (group.size >= n && strncmp(group.data, kShell[k], n) == 0)
                        currentShell = true;
                }
                for (size_t i = 0; i < visibility.size(); ++i) {
                    if (TokenEquals(group, visibility[i].group.c_str()))
                        hiddenCameras |= visibility[i].hiddenCameras;
                }
            }
        } else if (TokenEquals(kind, "v")) {
            Vertex v = {};
            TextToken x, y, z;
            if (!NextToken(line, &tokenOffset, &x) ||
                !NextToken(line, &tokenOffset, &y) ||
                !NextToken(line, &tokenOffset, &z) ||
                !ParseFloatToken(x, &v.x) || !ParseFloatToken(y, &v.y) ||
                !ParseFloatToken(z, &v.z) ||
                !ParseVertexColor(line, tokenOffset, &v) || vertices.size() >= kMaxVertices) {
                dbg_printf("[room3d] invalid OBJ vertex at line %u\n",
                           (unsigned int)lineNumber);
                return false;
            }
            vertices.push_back(v);
        } else if (TokenEquals(kind, "vt")) {
            Vertex uv = {};
            TextToken u, v;
            if (!NextToken(line, &tokenOffset, &u) ||
                !NextToken(line, &tokenOffset, &v) ||
                !ParseFloatToken(u, &uv.u) || !ParseFloatToken(v, &uv.v) ||
                texcoords.size() >= kMaxTexcoords) {
                dbg_printf("[room3d] invalid OBJ UV at line %u\n",
                           (unsigned int)lineNumber);
                return false;
            }
            texcoords.push_back(uv);
        } else if (TokenEquals(kind, "mtllib")) {
            TextToken libraryToken;
            if (!NextToken(line, &tokenOffset, &libraryToken)) continue;
            const std::string library = TokenString(libraryToken);
            if (!IsSafeRelativeAssetPath(library)) {
                dbg_printf("[room3d] unsafe material library path at line %u\n",
                           (unsigned int)lineNumber);
                continue;
            }
            LoadMaterialLibrary(objDirectory + library, &materials);
        } else if (TokenEquals(kind, "usemtl")) {
            TextToken materialToken;
            if (NextToken(line, &tokenOffset, &materialToken))
                currentMaterial = materialToken;
        } else if (TokenEquals(kind, "f")) {
            struct FaceIndex { size_t vertex; size_t texcoord; };
            std::vector<FaceIndex> face;
            TextToken token;
            while (NextToken(line, &tokenOffset, &token)) {
                FaceIndex index = { 0, static_cast<size_t>(-1) };
                if (!ParseFaceIndex(token, vertices.size(), texcoords.size(),
                                    &index.vertex, &index.texcoord)) {
                    dbg_printf("[room3d] invalid OBJ face at line %u\n",
                               (unsigned int)lineNumber);
                    return false;
                }
                face.push_back(index);
            }
            if (face.size() < 3) {
                dbg_printf("[room3d] face has fewer than 3 vertices at line %u\n",
                           (unsigned int)lineNumber);
                return false;
            }

            const int projectionCamera = hasProjection && faceIndex < projection.size()
                ? ProjectionCamera(projection[faceIndex]) : -1;
            ++faceIndex;
            const int materialIndex = GetMaterialIndex(currentMaterial, &materials);
            if (materialIndex < 0 || (size_t)materialIndex >= materials.size()) return false;
            materials[materialIndex].used = true;
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                if (triangles.size() >= kMaxTriangles) {
                    dbg_printf("[room3d] OBJ triangle limit exceeded\n");
                    return false;
                }
                Triangle tri = {};
                const FaceIndex indices[3] = { face[0], face[i], face[i + 1] };
                for (int v = 0; v < 3; ++v) {
                    tri.v[v] = vertices[indices[v].vertex];
                    if (indices[v].texcoord < texcoords.size()) {
                        tri.v[v].u = texcoords[indices[v].texcoord].u;
                        tri.v[v].v = texcoords[indices[v].texcoord].v;
                    }
                }
                tri.material = materialIndex;
                tri.hiddenCameras = hiddenCameras;
                tri.shell = currentShell;
                tri.projection = (signed char)projectionCamera;
                triangles.push_back(tri);
            }
        }
    }

    if (triangles.empty()) {
        dbg_printf("[room3d] OBJ has no triangles: %s\n", path);
        return false;
    }

    LoadMaterialTextures(objDirectory, &materials);

    // Projected faces: UVs are where each corner lands in its camera's
    // 320x240 frame (v flipped as Project flips it back); the lighting is in
    // the image, so the vertex color stays white.
    const bool projectionValid = hasProjection && faceIndex == projectionFaces;
    if (hasProjection && !projectionValid)
        dbg_printf("[room3d] %s.proj has %u faces, mesh has %u; ignoring it\n",
                   path, (unsigned int)projectionFaces, (unsigned int)faceIndex);
    if (projectionValid) {
        // Every in-room camera is loaded: the fixed cameras draw their own
        // background through the mesh, the free camera the assigned ones.
        const int count = g_RdtPointer != NULL ? (int)g_RdtPointer->cameras_count : 0;
        for (int k = 0; k < count && k < 32; ++k) {
            if ((s_backdropCameras & (1u << k)) != 0) continue;
            s_projectionTextures[k] = LoadBackgroundTexture(k);
            if (s_projectionTextures[k] == MARNI_NULL_HANDLE)
                dbg_printf("[room3d] no background for camera %d\n", k);
        }
        s_hasProjection = true;
    }

    unsigned int projectedCameras = 0;
    size_t projectedTriangles = 0;
    for (size_t i = 0; i < triangles.size(); ++i) {
        Triangle& tri = triangles[i];
        RoomCameraBasis basis;
        float focal = 0.0f;
        if (!projectionValid || tri.projection < 0 ||
            s_projectionTextures[tri.projection] == MARNI_NULL_HANDLE ||
            !RdtCameraBasis(tri.projection, &basis, &focal)) {
            tri.projection = -1;
            continue;
        }
        for (int v = 0; v < 3; ++v) {
            const float d[3] = { tri.v[v].x - basis.from[0], tri.v[v].y - basis.from[1],
                                 tri.v[v].z - basis.from[2] };
            const float x = basis.right[0] * d[0] + basis.right[1] * d[1] + basis.right[2] * d[2];
            const float y = basis.down[0] * d[0] + basis.down[1] * d[1] + basis.down[2] * d[2];
            float z = basis.forward[0] * d[0] + basis.forward[1] * d[1] + basis.forward[2] * d[2];
            if (z < 1.0f) z = 1.0f;
            tri.v[v].u = (x / z * focal + 160.0f) / 320.0f;
            tri.v[v].v = 1.0f - (y / z * focal + 120.0f) / 240.0f;
            tri.v[v].r = tri.v[v].g = tri.v[v].b = 1.0f;
        }
        projectedCameras |= 1u << tri.projection;
        ++projectedTriangles;
    }
    if (projectionValid)
        dbg_printf("[room3d] projection: %u of %u triangles from cameras 0x%x\n",
                   (unsigned int)projectedTriangles, (unsigned int)triangles.size(),
                   projectedCameras);
    for (size_t i = 0; i < triangles.size(); ++i) {
        if (triangles[i].projection >= 0) continue;
        Material& mat = materials[triangles[i].material];
        float r = mat.r, g = mat.g, b = mat.b;
        if (!mat.hasDiffuseColor && mat.texture == MARNI_NULL_HANDLE)
            MaterialColor(mat.name, &r, &g, &b);
        if (projectionValid) {
            for (size_t t = 0; t < tints.size(); ++t) {
                if (tints[t].material == mat.name) {
                    r *= tints[t].r; g *= tints[t].g; b *= tints[t].b;
                    break;
                }
            }
        }
        for (int v = 0; v < 3; ++v) {
            triangles[i].v[v].r *= r;
            triangles[i].v[v].g *= g;
            triangles[i].v[v].b *= b;
        }
    }

    s_triangles.swap(triangles);
    s_materials.swap(materials);
    s_loaded = true;
    dbg_printf("[room3d] loaded %s (%u triangles)\n", path,
               (unsigned int)s_triangles.size());
    return true;
}

float DepthNdc(float z)
{
    return TmdViewZToNdc(z);
}

bool BasisFromPoints(RoomCameraBasis* basis, const float from[3], const float to[3])
{
    const float dx = to[0] - from[0], dy = to[1] - from[1], dz = to[2] - from[2];
    const float length = sqrtf(dx * dx + dy * dy + dz * dz);
    const float horizontal = sqrtf(dx * dx + dz * dz);
    if (length < 1.0f || horizontal < 1.0f) return false;
    for (int i = 0; i < 3; ++i) basis->from[i] = from[i];
    basis->forward[0] = dx / length;
    basis->forward[1] = dy / length;
    basis->forward[2] = dz / length;
    basis->right[0] = dz / horizontal;
    basis->right[1] = 0.0f;
    basis->right[2] = -dx / horizontal;
    basis->down[0] = basis->forward[1] * basis->right[2];
    basis->down[1] = basis->forward[2] * basis->right[0] -
                     basis->forward[0] * basis->right[2];
    basis->down[2] = -basis->forward[1] * basis->right[0];
    return true;
}

bool BuildRoomCameraBasis(RoomCameraBasis* basis)
{
    // The modern follow camera, when it owns this tick, replaces the room's
    // fixed camera for the mesh exactly as it does for the actors.
    float modernFrom[3], modernTo[3];
    if (basis != NULL && ModernCamera_GetView(modernFrom, modernTo))
        return BasisFromPoints(basis, modernFrom, modernTo);
    if (basis == NULL || g_RdtPointer == NULL ||
        (unsigned int)g_roomCameraId >= (unsigned int)g_RdtPointer->cameras_count)
        return false;

    RDT_Camera* cameras =
        (RDT_Camera*)((char*)g_RdtPointer + sizeof(RDT));
    const RDT_Camera& camera = cameras[g_roomCameraId];
    // Match CollisionDebug.cpp's independent RDT projection basis. That path
    // is the engine's existing reference for putting raw room X/Y/Z onto the
    // same fixed-camera background and collision floor.
    const float dx = (float)(camera.cam_to_x - camera.cam_from_x);
    const float dy = (float)(camera.cam_to_y - camera.cam_from_y);
    const float dz = (float)(camera.cam_to_z - camera.cam_from_z);
    const float length = sqrtf(dx * dx + dy * dy + dz * dz);
    const float horizontal = sqrtf(dx * dx + dz * dz);
    if (length < 1.0f || horizontal < 1.0f || camera.roll != 0)
        return false;

    basis->from[0] = (float)camera.cam_from_x;
    basis->from[1] = (float)camera.cam_from_y;
    basis->from[2] = (float)camera.cam_from_z;
    basis->forward[0] = dx / length;
    basis->forward[1] = dy / length;
    basis->forward[2] = dz / length;
    basis->right[0] = dz / horizontal;
    basis->right[1] = 0.0f;
    basis->right[2] = -dx / horizontal;
    basis->down[0] = basis->forward[1] * basis->right[2];
    basis->down[1] = basis->forward[2] * basis->right[0] -
                     basis->forward[0] * basis->right[2];
    basis->down[2] = -basis->forward[1] * basis->right[0];
    return true;
}

bool Project(const Vertex& v, const RoomCameraBasis& camera,
             float scaleX, float scaleY, float centerX, float centerY,
             float focal, float nearZ, float* out)
{
    const float delta[3] = {
        v.x - camera.from[0], v.y - camera.from[1], v.z - camera.from[2]
    };
    float viewX = 0.0f, viewY = 0.0f, viewZ = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        viewX += camera.right[axis] * delta[axis];
        viewY += camera.down[axis] * delta[axis];
        viewZ += camera.forward[axis] * delta[axis];
    }
    // Clipped intersections are reconstructed in room coordinates, then this
    // dot product recomputes view Z. Floating-point roundoff can put an exact
    // near-plane point a few thousandths behind the plane; clamp that tiny
    // error instead of dropping the whole fan triangle.
    const float nearTolerance = 0.01f;
    if (viewZ < nearZ - nearTolerance) return false;
    if (viewZ < nearZ) viewZ = nearZ;

    const float inverseZ = focal / viewZ;
    out[0] = centerX + viewX * inverseZ * scaleX;
    out[1] = centerY + viewY * inverseZ * scaleY;
    out[2] = DepthNdc(viewZ);
    out[3] = viewZ;
    out[4] = v.u;
    out[5] = 1.0f - v.v;
    out[6] = v.r * s_activeShade;
    out[7] = v.g * s_activeShade;
    out[8] = v.b * s_activeShade;
    out[9] = 1.0f;
    return true;
}

float CameraDepth(const Vertex& vertex, const RoomCameraBasis& camera)
{
    return camera.forward[0] * (vertex.x - camera.from[0]) +
           camera.forward[1] * (vertex.y - camera.from[1]) +
           camera.forward[2] * (vertex.z - camera.from[2]);
}

Vertex InterpolateVertex(const Vertex& a, const Vertex& b, float t)
{
    Vertex result = {};
    result.x = a.x + (b.x - a.x) * t;
    result.y = a.y + (b.y - a.y) * t;
    result.z = a.z + (b.z - a.z) * t;
    result.u = a.u + (b.u - a.u) * t;
    result.v = a.v + (b.v - a.v) * t;
    result.r = a.r + (b.r - a.r) * t;
    result.g = a.g + (b.g - a.g) * t;
    result.b = a.b + (b.b - a.b) * t;
    return result;
}

int ClipTriangleToNearPlane(const Vertex input[3], const RoomCameraBasis& camera,
                            float nearZ, Vertex output[4])
{
    int outputCount = 0;
    for (int edge = 0; edge < 3; ++edge) {
        const Vertex& a = input[edge];
        const Vertex& b = input[(edge + 1) % 3];
        const float depthA = CameraDepth(a, camera);
        const float depthB = CameraDepth(b, camera);
        const bool insideA = depthA >= nearZ;
        const bool insideB = depthB >= nearZ;

        if (insideA && insideB) {
            output[outputCount++] = b;
        } else if (insideA != insideB) {
            const float t = (nearZ - depthA) / (depthB - depthA);
            output[outputCount++] = InterpolateVertex(a, b, t);
            if (insideB) output[outputCount++] = b;
        }
    }
    return outputCount;
}

} // namespace

bool Room3D_IsActive(void)
{
    // Opt-in transition telemetry for scripted runtime tests. This reports
    // observations, not pass/fail; the external runner asserts their order.
    static const bool trace = std::getenv("RE1_ROOM3D_TRACE") != NULL;
    if (trace) {
        const unsigned int screen = (unsigned int)g_main_state_flags &
            (MSF_ROOM_TRANSITION | MSF_MENU_ACTIVE | MSF_SCREEN_STANDALONE);
        static unsigned int lastScreen = ~0u;
        static int lastStage = -1, lastRoom = -1, lastCamera = -1;
        if (screen != lastScreen || lastStage != (int)g_stageId ||
            lastRoom != (int)g_roomId || lastCamera != (int)g_roomCameraId) {
            dbg_printf("[room3d-test] stage=%u room=%02X camera=%u menu=%u door=%u standalone=%u\n",
                (unsigned int)g_stageId + 1, (unsigned int)g_roomId,
                (unsigned int)g_roomCameraId,
                (screen & MSF_MENU_ACTIVE) ? 1u : 0u,
                (screen & MSF_ROOM_TRANSITION) ? 1u : 0u,
                (screen & MSF_SCREEN_STANDALONE) ? 1u : 0u);
            lastScreen = screen; lastStage = g_stageId;
            lastRoom = g_roomId; lastCamera = g_roomCameraId;
        }
        static int lastX = -100000, lastZ = -100000;
        const int x = (int)g_playerEntity.position.x;
        const int z = (int)g_playerEntity.position.z;
        if (std::abs(x - lastX) + std::abs(z - lastZ) >= 250) {
            dbg_printf("[room3d-position] room=%02X x=%d y=%d z=%d angle=%u\n",
                (unsigned int)g_roomId, x, (int)g_playerEntity.position.y, z,
                (unsigned int)(g_playerEntity.directionAngle & 0xfff));
            lastX = x; lastZ = z;
        }
    }

    // Door animations use their own camera and model scene while the next
    // RDT may already be loaded. Its room must not appear behind the door.
    // Inventory/item-box and standalone screens own their background as well;
    // drawing the room here would overwrite menu fills queued before 3D.
    if ((g_main_state_flags & (MSF_ROOM_TRANSITION | MSF_MENU_ACTIVE |
                               MSF_SCREEN_STANDALONE)) != 0) return false;
    if (!LoadRoomMesh() || g_sceneRenderParam <= 0 || Marni_DX() == NULL)
        return false;
    if (!ModernCamera_Active() && g_roomCameraId < 32 &&
        (s_backdropCameras & (1u << g_roomCameraId)) != 0)
        return false;
    RoomCameraBasis camera;
    return BuildRoomCameraBasis(&camera);
}

// The RDT's per-camera mask sprites exist so the player can pass behind
// objects painted into a flat backdrop. A room mesh occludes the player
// through the depth buffer instead, so drawing those sprites over it paints
// the original stairs and columns on top of the replacement geometry.
// RE1_ROOM3D_KEEP_MASKS=1 restores them for comparison.
bool Room3D_SuppressRoomMasks(void)
{
    static const bool keep = std::getenv("RE1_ROOM3D_KEEP_MASKS") != NULL;
    if (keep || !Room3D_IsActive()) return false;
    static int lastStage = -1, lastRoom = -1;
    if (lastStage != (int)g_stageId || lastRoom != (int)g_roomId) {
        dbg_printf("[room3d] room mask sprites suppressed for stage %u room %02X\n",
                   (unsigned int)g_stageId + 1, (unsigned int)g_roomId);
        lastStage = g_stageId; lastRoom = g_roomId;
    }
    return true;
}

void Room3D_Render(void)
{
    if (!Room3D_IsActive()) return;

    RoomCameraBasis camera;
    if (!BuildRoomCameraBasis(&camera)) return;
    const bool modern = ModernCamera_Active();
    s_activeShade = ((unsigned int)g_roomCameraId < 32 &&
                     (s_shadedCameras & (1u << g_roomCameraId)) != 0)
        ? s_cameraShade[g_roomCameraId] : 1.0f;

    float scaleX, scaleY;
    MarniGetRenderScale(&scaleX, &scaleY);
    const float centerX = (float)g_SubpixelOffsetX * scaleX;
    const float centerY = (float)g_SubpixelOffsetY * scaleY;
    const float focal = (float)g_sceneRenderParam;
    const float nearPlane = focal * 2.0f;
    float nearZ = nearPlane > kNearFloor ? nearPlane : kNearFloor;
    // When the follow camera's boom passes through a wall, everything nearer
    // than that wall is cut away so the shot always reaches the player.
    const float wallClip = ModernCamera_WallClipDepth();
    if (wallClip > nearZ) nearZ = wallClip;

    // A fixed camera with its background loaded draws every face with that
    // background, mapped by the face's own screen position: the original
    // art (or its HD replacement) with the mesh's depth behind it. The free
    // camera draws each assigned face from its own camera's background.
    const MarniHandle fixedBackground =
        (!modern && s_hasProjection && (unsigned int)g_roomCameraId < 32)
        ? s_projectionTextures[g_roomCameraId] : MARNI_NULL_HANDLE;

    float batch[kTrianglesPerBatch * 30];
    int batchCount = 0;
    MarniHandle batchTexture = MARNI_NULL_HANDLE;
    MarniSampler batchSampler = MARNI_SAMPLER_POINT;
    for (size_t i = 0; i < s_triangles.size(); ++i) {
        // Per-camera hides were authored for the fixed cameras; a free
        // camera sees the whole room.
        if (!modern && (unsigned int)g_roomCameraId < 32 &&
            (s_triangles[i].hiddenCameras & (1u << g_roomCameraId)) != 0)
            continue;
        if (modern && s_triangles[i].shell) {
            // A free camera can end up behind a wall; the shell faces into the
            // room, so drop the faces turned away from the viewer rather than
            // let a wall's back fill the screen.
            const Vertex* t = s_triangles[i].v;
            const float e1[3] = { t[1].x - t[0].x, t[1].y - t[0].y, t[1].z - t[0].z };
            const float e2[3] = { t[2].x - t[0].x, t[2].y - t[0].y, t[2].z - t[0].z };
            const float n[3] = { e1[1] * e2[2] - e1[2] * e2[1],
                                 e1[2] * e2[0] - e1[0] * e2[2],
                                 e1[0] * e2[1] - e1[1] * e2[0] };
            const float toEye[3] = { camera.from[0] - t[0].x, camera.from[1] - t[0].y,
                                     camera.from[2] - t[0].z };
            if (n[0] * toEye[0] + n[1] * toEye[1] + n[2] * toEye[2] < 0.0f) continue;
        }
        const int projection = s_triangles[i].projection;
        MarniHandle texture = s_materials[s_triangles[i].material].texture;
        MarniSampler sampler = MARNI_SAMPLER_POINT;
        if (fixedBackground != MARNI_NULL_HANDLE) {
            texture = fixedBackground;
        } else if (modern && projection >= 0 &&
                   s_projectionTextures[projection] != MARNI_NULL_HANDLE) {
            // Seen from a new angle the 320x240 art is magnified; filter it.
            texture = s_projectionTextures[projection];
            sampler = MARNI_SAMPLER_LINEAR;
        }
        // The camera sits inside the room, so large floor/wall triangles often
        // cross its near plane. Clip them in room space before perspective
        // projection; dropping a triangle with one behind-camera corner erased
        // the entire floor and left only distant wall fragments.
        Vertex clipped[4];
        const int clippedCount = ClipTriangleToNearPlane(
            s_triangles[i].v, camera, nearZ, clipped);

        for (int fan = 1; fan + 1 < clippedCount; ++fan) {
            const Vertex* triangle[3] = {
                &clipped[0], &clipped[fan], &clipped[fan + 1]
            };
            float vertices[30];
            bool visible = true;
            for (int v = 0; v < 3; ++v) {
                if (!Project(*triangle[v], camera, scaleX, scaleY,
                             centerX, centerY, focal, nearZ,
                             vertices + v * 10)) {
                    visible = false;
                    break;
                }
            }
            if (!visible) continue;
            if (fixedBackground != MARNI_NULL_HANDLE) {
                for (int v = 0; v < 3; ++v) {
                    float* out = vertices + v * 10;
                    out[4] = ((out[0] - centerX) / scaleX + 160.0f) / 320.0f;
                    out[5] = ((out[1] - centerY) / scaleY + 120.0f) / 240.0f;
                    out[6] = out[7] = out[8] = 1.0f;
                }
            }

            if (batchCount > 0 && (texture != batchTexture || sampler != batchSampler)) {
                Marni_DX()->DrawTriangles3D(batch, batchCount, batchTexture,
                                            batchSampler, MARNI_BLEND_DISABLE,
                                            true);
                batchCount = 0;
            }
            batchTexture = texture;
            batchSampler = sampler;

            memcpy(batch + batchCount * 30, vertices, sizeof(vertices));
            if (++batchCount == kTrianglesPerBatch) {
                Marni_DX()->DrawTriangles3D(batch, batchCount, batchTexture,
                                            batchSampler, MARNI_BLEND_DISABLE,
                                            true);
                batchCount = 0;
            }
        }
    }
    if (batchCount > 0) {
        Marni_DX()->DrawTriangles3D(batch, batchCount, batchTexture,
                                    batchSampler, MARNI_BLEND_DISABLE,
                                    true);
    }
}

// ---------------------------------------------------------------------------
// Queries for the modern follow camera (ModernCamera.cpp).

// True when this room has a usable mesh and the frame is not owned by a door,
// menu or standalone screen.
bool Room3D_MeshAvailable(void)
{
    if ((g_main_state_flags & (MSF_ROOM_TRANSITION | MSF_MENU_ACTIVE |
                               MSF_SCREEN_STANDALONE)) != 0) return false;
    return LoadRoomMesh() && Marni_DX() != NULL;
}

// Distance along `dir` (unit) from `origin` to the nearest mesh triangle, or
// `maxDistance` when nothing is hit first (two-sided Moller-Trumbore).
// `which`: 0 = every triangle, 1 = the room shell only, 2 = props only.
float Room3D_Raycast(const float origin[3], const float dir[3], float maxDistance,
                     int which)
{
    float best = maxDistance;
    for (size_t i = 0; i < s_triangles.size(); ++i) {
        if (which == 1 && !s_triangles[i].shell) continue;
        if (which == 2 && s_triangles[i].shell) continue;
        const Vertex* v = s_triangles[i].v;
        const float e1[3] = { v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z };
        const float e2[3] = { v[2].x - v[0].x, v[2].y - v[0].y, v[2].z - v[0].z };
        const float p[3] = { dir[1] * e2[2] - dir[2] * e2[1],
                             dir[2] * e2[0] - dir[0] * e2[2],
                             dir[0] * e2[1] - dir[1] * e2[0] };
        const float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (fabsf(det) < 1e-6f) continue;
        const float inv = 1.0f / det;
        const float t0[3] = { origin[0] - v[0].x, origin[1] - v[0].y, origin[2] - v[0].z };
        const float u = (t0[0] * p[0] + t0[1] * p[1] + t0[2] * p[2]) * inv;
        if (u < 0.0f || u > 1.0f) continue;
        const float q[3] = { t0[1] * e1[2] - t0[2] * e1[1],
                             t0[2] * e1[0] - t0[0] * e1[2],
                             t0[0] * e1[1] - t0[1] * e1[0] };
        const float w = (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]) * inv;
        if (w < 0.0f || u + w > 1.0f) continue;
        const float t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
        if (t > 1.0f && t < best) best = t;
    }
    return best;
}

// Headless 32-bit integration test for the opt-in ROOM1180 mesh path.
// Run from the repository root with the command in docs/3D_ROOM_PORT_PLAN.md.
#include "../src/game/Room3D.cpp"
#include "../src/system/PngImage.cpp"
#include <cassert>

BioCardLayout g_BioCard = {};
int g_GameMode = GAME_MODE_OG;
RDT* g_RdtPointer = NULL;
int g_sceneRenderParam = 0;
int g_SubpixelOffsetX = 160;
int g_SubpixelOffsetY = 120;
DWORD g_MainStateFlagBank[2] = { 0, 0 };

namespace {

MarniDX s_fakeDx;
int s_drawnTriangles = 0;
int s_floorTriangles = 0;
int s_createdTextures = 0;
int s_destroyedTextures = 0;
bool s_invalidVertex = false;
char s_requestedAssetPath[512] = {};

} // namespace

MarniDX::MarniDX() : m_pImpl(NULL) {}
MarniDX::~MarniDX() {}

MarniDX* Marni_DX()
{
    return &s_fakeDx;
}

MarniHandle MarniDX::CreateTexture(int width, int height, int bpp,
                                   const void* pixels, int*, int*)
{
    if (!((width == 256 && height == 128) || (width == 1254 && height == 1254)) ||
        bpp != 32 || pixels == NULL)
        s_invalidVertex = true;
    ++s_createdTextures;
    const unsigned char* rgba = (const unsigned char*)pixels;
    if (rgba != NULL && rgba[3] != 255) s_invalidVertex = true;
    return s_createdTextures;
}

void MarniDX::DestroyTexture(MarniHandle)
{
    ++s_destroyedTextures;
}

void MarniDX::DrawTriangles3D(const float* vertices, int triCount, MarniHandle texture,
                              MarniSampler, MarniBlend, bool)
{
    if (vertices == NULL || triCount <= 0) {
        s_invalidVertex = true;
        return;
    }
    s_drawnTriangles += triCount;
    for (int i = 0; i < triCount * 30; ++i) {
        if (!std::isfinite(vertices[i])) s_invalidVertex = true;
    }
    for (int triangle = 0; triangle < triCount; ++triangle) {
        bool isFloor = texture == 1;
        float minV = 1.0f;
        for (int vertex = 0; vertex < 3; ++vertex) {
            const float* v = vertices + triangle * 30 + vertex * 10;
            if (v[2] < 0.0f || v[2] > 1.0f ||
                v[3] < 2.0f * g_sceneRenderParam)
                s_invalidVertex = true;
            // The floor occupies the atlas's upper-left 64x64 cell. The wood
            // prop tile shares U but starts at V=0.5, so require an interior
            // floor V coordinate as well as the cell bounds.
            if (v[4] < -0.001f || v[4] > 0.251f ||
                v[5] < -0.001f || v[5] > 0.501f)
                isFloor = false;
            if (v[5] < minV) minV = v[5];
        }
        if (isFloor && minV < 0.49f) ++s_floorTriangles;
    }
}

void MarniGetRenderScale(float* scaleX, float* scaleY)
{
    *scaleX = 1.0f;
    *scaleY = 1.0f;
}

float TmdViewZToNdc(float viewZ)
{
    float depth = (viewZ - 1.0f) / (131072.0f - 1.0f);
    if (depth < 0.0f) depth = 0.0f;
    if (depth > 1.0f) depth = 1.0f;
    return depth;
}

extern "C" const char* ResolveAssetRoot(const char* requested, char* out, size_t size)
{
    // Keep the path passed by the room loader so tests can verify stage/room/
    // character naming without requiring a configured asset root.
    strncpy(s_requestedAssetPath, requested, sizeof(s_requestedAssetPath) - 1);
    s_requestedAssetPath[sizeof(s_requestedAssetPath) - 1] = '\0';
    const char* path = strstr(requested, "room1181.obj")
        ? "tests/fixtures/room3d/Stage1/room1181.obj"
        : "tests/fixtures/room3d/Stage1/room1180.obj";
    if (strlen(path) + 1 > size) return "";
    memcpy(out, path, strlen(path) + 1);
    return out;
}

unsigned char room_file_stage(void)
{
    return (unsigned char)g_stageId;
}

const char* plat_normalize_path(const char* path, char* out, size_t size)
{
    const size_t length = strlen(path);
    if (length + 1 > size) return path;
    memcpy(out, path, length + 1);
    return out;
}

BOOL plat_is_debugger_present(void) { return FALSE; }
void plat_debug_output(const char*) {}
DWORD plat_env_get(const char*, char*, DWORD) { return 0; }

int main()
{
    const char* colorInputs[] = {"", "1", "0.25 0.5 0.75", "0 1", "nan 0 1",
                                 "0 0 1.1", "0 0 -0.1", "0 0 1 1"};
    for (size_t i = 0; i < sizeof(colorInputs)/sizeof(colorInputs[0]); ++i) {
        Vertex vertex = {};
        const TextToken tail = {colorInputs[i], strlen(colorInputs[i])};
        assert(ParseVertexColor(tail, 0, &vertex) == (i < 3));
        if (i < 2) assert(vertex.r == 1 && vertex.g == 1 && vertex.b == 1);
        if (i == 2) assert(vertex.r == .25f && vertex.g == .5f && vertex.b == .75f);
    }
    Vertex dark = {}, bright = {};
    bright.r = 1; bright.g = .5f; bright.b = .25f;
    const Vertex midpoint = InterpolateVertex(dark, bright, .5f);
    assert(midpoint.r == .5f && midpoint.g == .25f && midpoint.b == .125f);

    struct RoomAndCamera {
        RDT room;
        RDT_Camera cameras[2];
    } data = {};
    data.room.cameras_count = 2;
    data.cameras[0].cam_from_x = 2682;
    data.cameras[0].cam_from_y = -1836;
    data.cameras[0].cam_from_z = 1872;
    data.cameras[0].cam_to_x = 4950;
    data.cameras[0].cam_to_y = -2628;
    data.cameras[0].cam_to_z = 9018;
    data.cameras[0].fov = 207;
    data.cameras[1].cam_from_x = 2502;
    data.cameras[1].cam_from_y = -1800;
    data.cameras[1].cam_from_z = 9900;
    data.cameras[1].cam_to_x = 4950;
    data.cameras[1].cam_to_y = -2610;
    data.cameras[1].cam_to_z = 2088;
    data.cameras[1].fov = 207;

    g_RdtPointer = &data.room;
    g_stageId = STAGE_MANSION_1F;
    g_roomId = ROOM_MANSION_STOREROOM;
    g_roomCameraId = 0;
    g_sceneRenderParam = data.cameras[0].fov;

    assert(Room3D_IsActive());
    assert(s_createdTextures == 2);
    int sourceFloorTriangles = 0;
    for (size_t triangle = 0; triangle < s_triangles.size(); ++triangle) {
        bool usesFloorAtlasCell =
            strcmp(s_materials[s_triangles[triangle].material].name, "room") == 0;
        for (int vertex = 0; vertex < 3; ++vertex) {
            const Vertex& v = s_triangles[triangle].v[vertex];
            if (v.u < -0.001f || v.u > 0.251f || v.v < 0.499f)
                usesFloorAtlasCell = false;
        }
        if (usesFloorAtlasCell) {
            ++sourceFloorTriangles;
            for (int vertex = 0; vertex < 3; ++vertex)
                assert(s_triangles[triangle].v[vertex].y == 0.0f);
        }
    }
    assert(sourceFloorTriangles > 0);
    Room3D_Render();
    assert(s_drawnTriangles > 0);
    assert(s_floorTriangles > 0);
    const int camera0Triangles = s_drawnTriangles;
    const int camera0FloorTriangles = s_floorTriangles;

    g_roomCameraId = 1;
    g_sceneRenderParam = data.cameras[1].fov;
    assert(Room3D_IsActive());
    Room3D_Render();
    assert(s_drawnTriangles > camera0Triangles);
    assert(s_floorTriangles > camera0FloorTriangles);
    // Foreground cutaways affect only the authored fixed view.
    const std::vector<Triangle> completeMesh = s_triangles;
    s_triangles.clear();
    for (size_t i = 0; i < completeMesh.size(); ++i)
        if ((completeMesh[i].hiddenCameras & 2u) != 0)
            s_triangles.push_back(completeMesh[i]);
    assert(!s_triangles.empty());
    const int beforeCutaway = s_drawnTriangles;
    Room3D_Render();
    assert(s_drawnTriangles == beforeCutaway);
    g_roomCameraId = 0;
    Room3D_Render();
    assert(s_drawnTriangles > beforeCutaway);
    s_triangles = completeMesh;
    assert(!s_invalidVertex);

    // Jill's explicit asset must load too; no implicit cross-variant fallback.
    g_main_state_flags |= MSF_CHAR_VARIANT;
    assert(Room3D_IsActive());
    assert(strstr(s_requestedAssetPath, "Stage1/room1181.obj") != NULL);
    const int beforeJill = s_drawnTriangles;
    Room3D_Render();
    assert(s_drawnTriangles > beforeJill);
    assert(s_createdTextures == 4 && s_destroyedTextures == 2);

    // A door transition must select a separate mesh for the new room, and
    // switching character variants must select that room's variant asset.
    g_stageId = STAGE_MANSION_2F;
    g_roomId = ROOM_ARMOR_ROOM;
    g_main_state_flags |= MSF_CHAR_VARIANT;
    assert(Room3D_IsActive());
    assert(strstr(s_requestedAssetPath, "Stage2/room2051.obj") != NULL);
    assert(s_destroyedTextures == 4);
    assert(s_createdTextures == 6);
    const int beforeDoor = s_drawnTriangles;
    g_main_state_flags |= MSF_ROOM_TRANSITION;
    assert(!Room3D_IsActive());
    Room3D_Render();
    assert(s_drawnTriangles == beforeDoor);
    g_main_state_flags &= ~MSF_ROOM_TRANSITION;
    assert(Room3D_IsActive());
    for (unsigned int screen : {MSF_MENU_ACTIVE, MSF_SCREEN_STANDALONE}) {
        const int beforeMenu = s_drawnTriangles;
        g_main_state_flags |= screen;
        assert(!Room3D_IsActive());
        Room3D_Render();
        assert(s_drawnTriangles == beforeMenu);
        g_main_state_flags &= ~screen;
        assert(Room3D_IsActive());
        Room3D_Render();
        assert(s_drawnTriangles > beforeMenu);
    }
    assert(s_createdTextures == 6); // UI transitions keep the cached room.
    return 0;
}

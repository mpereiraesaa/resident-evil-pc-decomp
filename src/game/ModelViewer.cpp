// ModelViewer.cpp - Port-only character inspection mode.
//
// RE1_MODEL_VIEW=<name> starts this task after load_global_assets. Geometry,
// textures, joint setup, lighting and drawing all go through the game's own
// EMD/TMD functions. RE1_MODEL_VIEW_SCENE=main-hall also loads the real room.
#include "../Globals.h"
#include "Entities.h"
#include "../platform/platform.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern void InitPlayerEntity(void);              // 0x004950b0
extern void LoadEntityEMD(Entity*, unsigned char); // 0x00462370
extern void Entity_SetJoints(Entity*, unsigned int); // 0x0048bc60
extern void InitAnimStructure(void*);            // 0x0048b6b0
extern unsigned int SetupJointStructures(unsigned int); // 0x0048b9e0
extern void ResetJointTransforms(void);          // 0x0048bad0
extern void EntityComputeJointWorldMatrices(int); // 0x0048c190
extern void EntityApplyLookAtRotation(void);     // 0x0045a2e0
extern void SetSubpixelOffset(int, int);         // 0x00470a30
extern void SetVideoResolution(int, int);         // 0x00497f30
extern void setSomeColor(int, int, int);           // 0x00470a50
extern bool EntityMeshSourceInfo(unsigned int, const char**, unsigned int*); // port-only

struct PreviewModel {
    const char* name;
    unsigned char index; // g_emdPathTable index in EntityModelLoader.cpp
    unsigned char type;  // entity id used by the ordinary renderer
    unsigned char player; // Chris/Jill model-table half
};

// Indices and filenames come from g_emdPathTable (0x004c1320). The first
// four files are player models; indices 36+ are room actor models.
static const PreviewModel kPreviewModels[] = {
    {"chris", 0, 0, 0}, {"jill", 1, 1, 1},
    {"barry", 38, 34, 0}, {"rebecca", 39, 35, 0},
    {"wesker", 40, 36, 0},
    {"zombie", 4, 0, 0}, {"zombie-jill", 4, 0, 1},
    {"naked-zombie", 5, 1, 0}, {"naked-zombie-jill", 5, 1, 1},
    {"green-zombie", 21, 17, 0}, {"green-zombie-jill", 21, 17, 1},
    {"cerberus", 6, 2, 0}, {"hunter", 10, 6, 0},
    {"tyrant", 16, 12, 0},
};

static int PreviewCoordinate(const char* name, int fallback)
{
    const char* value = getenv(name);
    if (value == NULL) return fallback;
    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < -32768 || parsed > 32767) {
        fprintf(stderr, "RE1_MODEL_VIEW: invalid %s='%s'\n", name, value);
        exit(2);
    }
    return (int)parsed;
}

static void ExportPreviewRig(const char* requested)
{
    const char* directory = getenv("RE1_ENTITY_OBJ_EXPORT_DIR");
    if (directory == NULL || directory[0] == 0 || ENTITY->jointsStructs == NULL ||
        ENTITY->animHeader == 0) return;
    char stem[64];
    size_t length = strlen(requested);
    if (length >= sizeof(stem)) length = sizeof(stem) - 1;
    for (size_t i = 0; i < length; ++i) {
        const char c = requested[i];
        stem[i] = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') ? c : '_';
    }
    stem[length] = 0;
    char filename[1024];
    const int pathLength = snprintf(filename, sizeof(filename), "%s/%s_rig.json",
                                    directory, stem);
    if (pathLength < 0 || pathLength >= (int)sizeof(filename)) return;
    FILE* out = fopen(filename, "wb");
    if (out == NULL) {
        fprintf(stderr, "RE1_ENTITY_OBJ_EXPORT: cannot create %s\n", filename);
        return;
    }
    const int count = (int)ENTITY->jointCount;
    const char* sourcePath = NULL;
    EntityMeshSourceInfo(ENTITY->modelLoadBuffer, &sourcePath, NULL);
    if (sourcePath == NULL) sourcePath = "unknown";
    int parent[256];
    for (int i = 0; i < 256; ++i) parent[i] = -1;
    const unsigned int baseOffset = *(unsigned short*)ENTITY->animHeader & ~3u;
    const unsigned char* tree = (const unsigned char*)ENTITY->animHeader + baseOffset;
    for (int i = 0; i < count; ++i) {
        const int childCount = tree[i * 4];
        const unsigned int listOffset = *(const unsigned short*)(tree + i * 4 + 2);
        const unsigned char* children = tree + listOffset;
        for (int c = 0; c < childCount; ++c)
            if (children[c] < count) parent[children[c]] = i;
    }
    const char* stage = getenv("RE1_ENTITY_OBJ_EXPORT_STAGE");
    if (stage == NULL || (strcmp(stage, "refined") != 0 &&
                          strcmp(stage, "patched") != 0)) stage = "source";
    fprintf(out, "{\n  \"model\": \"%s\",\n"
            "  \"sourcePath\": \"%s\",\n  \"jointCount\": %d,\n"
            "  \"coordinateSystem\": \"TMD viewport local, Y up; rest translations are original GTE units\",\n"
            "  \"joints\": [\n", stem, sourcePath, count);
    for (int i = 0; i < count; ++i) {
        const JointStruct* joint = &ENTITY->jointsStructs[i];
        const unsigned int* animObject = (const unsigned int*)joint->anim_object;
        const char* emdPath = NULL;
        unsigned int offset = 0;
        const bool hasMesh = animObject != NULL &&
            EntityMeshSourceInfo(animObject[0], &emdPath, &offset);
        fprintf(out, "    {\"index\": %d, \"parent\": %d, "
                "\"restTranslation\": [%d, %d, %d], \"obj\": ",
                i, parent[i], joint->transform.t[0], joint->transform.t[1],
                joint->transform.t[2]);
        bool exported = false;
        char objName[128];
        if (hasMesh) {
            const char* base = strrchr(emdPath, '/');
            base = base ? base + 1 : emdPath;
            size_t baseLength = strlen(base);
            if (baseLength >= 4 && strcmp(base + baseLength - 4, ".emd") == 0)
                baseLength -= 4;
            const int nameLength = snprintf(objName, sizeof(objName),
                                            "%.*s_%08x_%s.obj",
                                            (int)baseLength, base, offset, stage);
            if (nameLength > 0 && nameLength < (int)sizeof(objName)) {
                char objPath[1024];
                const int fullLength = snprintf(objPath, sizeof(objPath), "%s/%s",
                                                directory, objName);
                if (fullLength > 0 && fullLength < (int)sizeof(objPath)) {
                    FILE* exportedFile = fopen(objPath, "rb");
                    if (exportedFile != NULL) {
                        exported = true;
                        fclose(exportedFile);
                    }
                }
            }
        }
        if (exported)
            fprintf(out, "\"%s\"", objName);
        else
            fprintf(out, "null");
        fprintf(out, "}%s\n", i + 1 < count ? "," : "");
    }
    fprintf(out, "  ]\n}\n");
    fclose(out);
    fprintf(stderr, "RE1_ENTITY_OBJ_EXPORT: %s\n", filename);
}

// Port-only preview task; no address in ResidentEvil.exe.
void ModelViewer_Run(void)
{
    const char* requested = getenv("RE1_MODEL_VIEW");
    const PreviewModel* model = NULL;
    PreviewModel indexedModel = {};
    for (unsigned int i = 0; i < sizeof(kPreviewModels) / sizeof(kPreviewModels[0]); ++i) {
        if (requested != NULL && strcmp(requested, kPreviewModels[i].name) == 0) {
            model = &kPreviewModels[i];
            break;
        }
    }
    // The retail path table has 53 entries per character block. Numeric
    // selection makes every original entry inspectable, including variants
    // that have no friendly alias above. Prefix with "j:" for Jill's block.
    if (model == NULL && requested != NULL) {
        const bool jillBlock = requested[0] == 'j' && requested[1] == ':';
        const char* number = jillBlock ? requested + 2 : requested;
        char* end = NULL;
        long index = strtol(number, &end, 0);
        if (end != number && *end == '\0' && index >= 0 && index < 53) {
            indexedModel.name = requested;
            indexedModel.index = (unsigned char)index;
            indexedModel.type = (unsigned char)(index < 4 ? index : index - 4);
            indexedModel.player = jillBlock ? 1 : 0;
            model = &indexedModel;
        }
    }
    if (model == NULL) {
        fprintf(stderr, "RE1_MODEL_VIEW: unknown model '%s'\n", requested ? requested : "");
        fprintf(stderr, "Available:");
        for (unsigned int i = 0; i < sizeof(kPreviewModels) / sizeof(kPreviewModels[0]); ++i)
            fprintf(stderr, " %s", kPreviewModels[i].name);
        fprintf(stderr, "\n");
        exit(2);
        return;
    }

    const char* scene = getenv("RE1_MODEL_VIEW_SCENE");
    const bool mainHall = scene != NULL && strcmp(scene, "main-hall") == 0;
    if (scene != NULL && !mainHall) {
        fprintf(stderr, "RE1_MODEL_VIEW_SCENE: unknown scene '%s' (available: main-hall)\n", scene);
        exit(2);
        return;
    }
    // Camera 4 looks down the upper hall from behind the staircase.
    // Camera 0 is the distant overhead comparison view.
    int hallCamera = 4;
    const char* cameraName = getenv("RE1_MODEL_VIEW_CAMERA");
    if (mainHall && cameraName != NULL) {
        char* end = NULL;
        long selected = strtol(cameraName, &end, 10);
        if (end == cameraName || *end != '\0' || selected < 0 || selected > 7) {
            fprintf(stderr, "RE1_MODEL_VIEW_CAMERA: expected main hall camera 0..7\n");
            exit(2);
            return;
        }
        hallCamera = (int)selected;
    }

    // EMDs are larger than the retail player buffer. Keep the file, resolved
    // animation data and TMD objects together in a private, aligned work area.
    const unsigned int workSize = 2 * 1024 * 1024;
    void* work = malloc(workSize);
    if (work == NULL) {
        fprintf(stderr, "RE1_MODEL_VIEW: out of memory\n");
        exit(2);
        return;
    }
    memset(work, 0, workSize);

    if (mainHall) {
        // The retail room loader relocates pointers in place. Keep its RDT in
        // a separate buffer so EMD and joint allocation cannot overwrite it.
        void* room = malloc(2 * 1024 * 1024);
        if (room == NULL) {
            fprintf(stderr, "RE1_MODEL_VIEW: out of room memory\n");
            exit(2);
            return;
        }
        memset(room, 0, 2 * 1024 * 1024);
        g_stageId = STAGE_MANSION_1F;
        g_roomId = ROOM_MAIN_HALL;
        g_roomCameraId = (unsigned char)hallCamera;
        if (model->player)
            g_main_state_flags |= MSF_CHAR_VARIANT;
        else
            g_main_state_flags &= ~MSF_CHAR_VARIANT;
        g_loadDataDestPointer = room;
        LoadRoomRdt();                 // 0x00477d90; sets ambient back colour
    }

    ENTITY = (Entity*)&g_playerEntity;
    InitPlayerEntity();
    // Keep the enemy type for LoadEntityEMD's texture queue and model setup.
    // The viewer's Chris/Jill asset block is selected separately below.
    g_playerEntity.id = model->type;
    g_playerEntity.zoneFlags = 1;
    g_playerEntity.scaMatrixData.localMatrix = g_identityMatrixData;
    // Upper-hall camera 4 does not see the ground-floor starting position.
    // Place the preview on its gallery floor, facing toward that camera.
    const bool upperHall = mainHall && hallCamera == 4;
    g_playerEntity.scaMatrixData.localMatrix.t[0] =
        PreviewCoordinate("RE1_MODEL_VIEW_X", upperHall ? 16000 : mainHall ? 17000 : 0);
    g_playerEntity.scaMatrixData.localMatrix.t[1] =
        PreviewCoordinate("RE1_MODEL_VIEW_Y", upperHall ? 0 : mainHall ? 0 : 0x834);
    g_playerEntity.scaMatrixData.localMatrix.t[2] =
        PreviewCoordinate("RE1_MODEL_VIEW_Z", upperHall ? 23500 : mainHall ? 5000 : -700);
    g_playerEntity.position.x = g_playerEntity.scaMatrixData.localMatrix.t[0];
    g_playerEntity.position.y = g_playerEntity.scaMatrixData.localMatrix.t[1];
    g_playerEntity.position.z = g_playerEntity.scaMatrixData.localMatrix.t[2];
    g_playerEntity.directionAngle =
        PreviewCoordinate("RE1_MODEL_VIEW_ANGLE", upperHall ? 0 : mainHall ? 3072 : 0xce4) & 0xfff;

    g_TextureCurrentPage = 7;
    g_TextureBankID = 0x16;
    g_loadDataDestPointer = work;
    if (model->player)
        g_main_state_flags |= MSF_CHAR_VARIANT;
    else
        g_main_state_flags &= ~MSF_CHAR_VARIANT;
    const bool traceLoad = getenv("RE1_MODEL_VIEW_TRACE") != NULL;
    if (getenv("RE1_HD_TEXTURE_AUDIT") != NULL)
        fprintf(stderr, "RE1_MODEL_VIEW_EMD_BEGIN: %s\n", model->name);
    LoadEntityEMD(ENTITY, model->index);
    if (traceLoad) fprintf(stderr, "RE1_MODEL_VIEW_STAGE: EMD loaded\n");
    g_playerEntity.id = model->type;
    Entity_SetJoints(ENTITY, sizeof(JointStruct));
    if (traceLoad) fprintf(stderr, "RE1_MODEL_VIEW_STAGE: joints allocated\n");
    InitAnimStructure((void*)ENTITY->modelLoadBuffer);
    if (traceLoad) fprintf(stderr, "RE1_MODEL_VIEW_STAGE: animation initialized\n");
    g_loadDataDestPointer = (void*)SetupJointStructures((unsigned int)g_loadDataDestPointer);
    if (traceLoad) fprintf(stderr, "RE1_MODEL_VIEW_STAGE: joint structures ready\n");
    ResetJointTransforms();
    if (traceLoad) fprintf(stderr, "RE1_MODEL_VIEW_STAGE: transforms reset\n");
    ExportPreviewRig(requested);
    // The viewer uses page 7 for drawing. SetupTextureBankData subtracts 10
    // from this argument when any EMD queues textures; passing 7 makes its
    // unsigned copy count wrap and stalls on creatures with queued pages.
    // Ten performs the queue setup without reserving extra bank cells.
    SetupTextureBankData(10);
    if (traceLoad) fprintf(stderr, "RE1_MODEL_VIEW_STAGE: texture bank ready\n");

    if (mainHall) {
        // room_set loads art after its entity textures; keep that same order.
        load_room_bg();                // 0x00462b00; original camera PAKs
        load_room_bg_image();          // 0x004629c0; activate chosen camera
        setSomeColor(128, 128, 128);   // 0x00470a50; neutral background tint
    }

    g_lightMatrix = g_identityMatrixData;
    if (mainHall) {
        Room_SetupCamera();           // 0x00462970; RDT camera and FOV
        // render_entity calls update_entity_lighting with the RDT lamps.
    } else {
        // Match the options screen's standalone camera and three white lamps.
        MATRIX menuCamera = {};
        menuCamera.m[1][1] = 5000;
        MatrixToCamera(&menuCamera);
        int light[4] = { 100, 80, -780, 0x00808080 };
        FUN_0040ac80(0, light);
        light[0] = -100;
        FUN_0040ac80(1, light);
        light[1] = 0;
        FUN_0040ac80(2, light);
        setBackColor(409, 409, 409);
        set_scene_render_param(0xcf);
    }
    // Viewer-only optical zoom for small close-up EMDs such as the typing
    // hands. This uses the engine's normal projection path and leaves model
    // transforms, joints, room lights, and texture loading unchanged.
    if (getenv("RE1_MODEL_VIEW_FOV") != NULL) {
        const int fov = PreviewCoordinate("RE1_MODEL_VIEW_FOV", 0);
        if (fov < 64 || fov > 512) {
            fprintf(stderr, "RE1_MODEL_VIEW_FOV: expected 64..512\n");
            exit(2);
        }
        set_scene_render_param(fov);
    }
    SetVideoResolution(320, 240);
    SetSubpixelOffset(160, 120);
    g_main_state_flags = (g_main_state_flags & ~MSF_SCREEN_MODE_MASK) |
        (mainHall ? MSF_SCREEN_REBUILD : MSF_SCREEN_STANDALONE);
    g_spriteAnimR = 20;
    g_spriteAnimG = 24;
    g_spriteAnimB = 30;

    const int playerAnim = PreviewCoordinate("RE1_MODEL_VIEW_PLAYER_ANIM", -1);
    if (playerAnim >= 0 && (playerAnim > 31 || model->index >= 4)) {
        fprintf(stderr, "RE1_MODEL_VIEW_PLAYER_ANIM: requires a player model "
                        "and animation index 0..31\n");
        exit(2);
    }
    if (playerAnim >= 0) {
        // Joint_move reads the Entity layout at +0xBD; in PlayerEntity this
        // byte is named attackAnim, while animationId at +0x84 is a different
        // player state field.
        ENTITY->animationId = (unsigned char)playerAnim;
        g_playerEntity.animation_frame_id = 0;
        ENTITY->timing_control = 0;
    }

    const bool stillStudio = !mainHall && getenv("RE1_MODEL_VIEW_STILL") != NULL;
    fprintf(stderr, "RE1_MODEL_VIEW: %s (EMD index %u), ",
            model->name, (unsigned)model->index);
    if (mainHall)
        fprintf(stderr, "main hall camera %d\n", hallCamera);
    else if (stillStudio)
        fprintf(stderr, "studio still angle\n");
    else
        fprintf(stderr, "rotating at 30 Hz\n");
    const int exitFrames = PreviewCoordinate("RE1_MODEL_VIEW_EXIT_FRAMES", 0);
    const int clones = PreviewCoordinate("RE1_MODEL_VIEW_CLONES", 1);
    if (clones < 1 || clones > 32) {
        fprintf(stderr, "RE1_MODEL_VIEW_CLONES: expected 1..32\n");
        exit(2);
    }
    // Hold a repeatable pose while comparing mesh or lighting variants. The
    // chosen frame counts actual Joint_move calls, starting at zero.
    const int freezeAfter = PreviewCoordinate("RE1_MODEL_VIEW_FREEZE_AFTER", -1);
    int renderedFrames = 0;
    for (;;) {
        ENTITY = (Entity*)&g_playerEntity;
        if (!mainHall && !stillStudio)
            g_playerEntity.directionAngle = (g_playerEntity.directionAngle + 12) & 0xfff;
        // Room enemies and NPCs animate from the EMD's own EMR/EDD pair.
        // The player CHAR EMD carries its ordinary motion pair. The optional
        // preview plays that same pair as the game's player animation path.
        if ((model->index >= 4 || playerAnim >= 0) &&
            (freezeAfter < 0 || renderedFrames < freezeAfter))
            Joint_move(0, ENTITY->animHeader, ENTITY->animBase, 0x400);
        if (traceLoad && playerAnim >= 0 && renderedFrames < 12)
            fprintf(stderr, "RE1_MODEL_VIEW_ANIM: tick=%d frame=%u joint1=(%d,%d,%d)\n",
                    renderedFrames, (unsigned)ENTITY->animation_frame_id,
                    (int)ENTITY->jointsStructs[1].rotation.x,
                    (int)ENTITY->jointsStructs[1].rotation.y,
                    (int)ENTITY->jointsStructs[1].rotation.z);
        EntityComputeJointWorldMatrices(0);
        EntityApplyLookAtRotation();
        // Identical draws stress the renderer's per-frame triangle capacity
        // without altering the EMD, animation, or the character placement.
        for (int clone = 0; clone < clones; ++clone)
            render_entity(ENTITY);
        ++renderedFrames;
        if (exitFrames > 0 && renderedFrames >= exitFrames)
            exit(0);
        Task_sleep(1);
    }
}

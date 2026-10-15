// ModernCamera.cpp - port addition: a third-person follow camera for rooms
// that have a Room3D mesh, in the style of the modern Resident Evil games.
//
// The original game picks one of a room's fixed RDT cameras by zone and draws
// a pre-rendered background behind the actors. Once a room is real geometry
// (Room3D) any viewpoint works, so this module places the camera behind and
// above the player every tick and installs it through the same path the fixed
// cameras use (MatrixToCamera -> g_RoomCameraData), which the actors, the room
// mesh and the camera-relative analogue movement all read.
//
// It steps aside (the fixed camera returns) whenever the room has no mesh,
// during door and room transitions, while a menu or standalone screen owns the
// frame, when the player is dead, and while a script holds the camera
// (MSF_CAMERA_LOCK), so scripted cutscenes keep their authored shots.
//
// Controls: the right stick orbits (X) and tilts (Y). With tank controls
// (keyboard / D-pad) the camera eases back behind the player while moving.
// When a wall stands between the player and the camera, the camera keeps its
// distance and the room renderer clips away everything nearer than that wall,
// so the shot always reaches the player.
//
// Enabled by [Game] ModernCamera=1 or RE1_MODERN_CAMERA=1.
#include "../Globals.h"
#include "../DebugPrint.h"
#include "../marni/MarniXInput.h"
#include <cmath>
#include <cstdlib>

bool  Room3D_MeshAvailable(void);                                       // Room3D.cpp
float Room3D_Raycast(const float origin[3], const float dir[3], float maxDistance); // Room3D.cpp

bool g_bModernCamera = false;   // [Game] ModernCamera

namespace {

const float kPi = 3.14159265358979f;
const float kEyeHeight = 1650.0f;      // pivot above the player's feet (room units, -Y up)
const float kShoulder = 380.0f;        // over-the-right-shoulder offset
const float kDistance = 2700.0f;       // boom length
const float kWallMargin = 220.0f;
const float kDefaultPitch = 0.20f;     // radians, positive looks down
const float kMinPitch = -0.45f;
const float kMaxPitch = 0.95f;
const float kYawSpeed = 3.0f / 30.0f;  // radians per tick at full stick
const float kPitchSpeed = 1.8f / 30.0f;
const int   kFocal = 220;              // ~57 degrees vertical field of view

bool  s_active = false;
bool  s_wasActive = false;
bool  s_valid = false;
float s_yaw = 0.0f;
float s_pitch = kDefaultPitch;
float s_distance = kDistance;
float s_from[3], s_to[3];
int   s_lastStage = -1, s_lastRoom = -1;
bool  s_justEntered = true;
float s_wallClip = 0.0f;
int   s_lastLoggedRoom = -1;

bool Enabled(void)
{
    static const bool env = std::getenv("RE1_MODERN_CAMERA") != NULL;
    return g_bModernCamera || env;
}

float PlayerHeading(void)
{
    // directionAngle is 12-bit; heading t moves the player along (cos t, -sin t).
    return (float)(g_playerEntity.directionAngle & 0xfff) * (2.0f * kPi / 4096.0f);
}

float WrapAngle(float a)
{
    while (a > kPi) a -= 2.0f * kPi;
    while (a < -kPi) a += 2.0f * kPi;
    return a;
}

bool Allowed(void)
{
    if (!Enabled()) return false;
    if ((g_main_state_flags & (MSF_ROOM_TRANSITION | MSF_DOOR_TRANSITION |
                               MSF_CAMERA_LOCK | MSF_MENU_ACTIVE |
                               MSF_SCREEN_STANDALONE | MSF_PLAYER_DEAD)) != 0)
        return false;
    return Room3D_MeshAvailable();
}

} // namespace

bool ModernCamera_Active(void) { return s_active; }

float ModernCamera_WallClipDepth(void) { return s_active ? s_wallClip : 0.0f; }

bool ModernCamera_GetView(float from[3], float to[3])
{
    if (!s_active || !s_valid) return false;
    for (int i = 0; i < 3; ++i) { from[i] = s_from[i]; to[i] = s_to[i]; }
    return true;
}

// Called once per game tick after the room's camera-zone check, before the
// actors are transformed.
void ModernCamera_Update(void)
{
    s_active = Allowed();
    if (!s_active) {
        if (s_wasActive) {
            // Hand the frame back to the room's own fixed camera.
            Room_SetupCamera();
            dbg_printf("[modern-camera] fixed camera restored (stage %u room %02X)\n",
                       (unsigned int)g_stageId + 1, (unsigned int)g_roomId);
        }
        s_wasActive = false;
        s_valid = false;
        return;
    }

    const bool newRoom = (int)g_stageId != s_lastStage || (int)g_roomId != s_lastRoom;
    if (!s_wasActive || newRoom) {
        // Start behind the player on entry.
        s_yaw = PlayerHeading();
        s_pitch = kDefaultPitch;
        s_distance = kDistance;
        s_justEntered = true;
        s_lastStage = g_stageId;
        s_lastRoom = g_roomId;
        dbg_printf("[modern-camera] active in stage %u room %02X\n",
                   (unsigned int)g_stageId + 1, (unsigned int)g_roomId);
    }

    // Right stick: orbit and tilt.
    float rx = 0.0f, ry = 0.0f, rmag = 0.0f;
    const bool stick = MarniXInput::GetRightStick(&rx, &ry, &rmag) && rmag > 0.0f;
    if (stick) {
        s_yaw = WrapAngle(s_yaw - rx * kYawSpeed);
        s_pitch -= ry * kPitchSpeed;
        if (s_pitch < kMinPitch) s_pitch = kMinPitch;
        if (s_pitch > kMaxPitch) s_pitch = kMaxPitch;
    }

    // Tank controls (keyboard, D-pad, or analogue movement switched off):
    // ease back behind the player while walking or running.
    const bool tank = !g_bAlternateControls || MarniXInput::IsDpadActive() ||
                      !MarniXInput::GetLeftStick(NULL, NULL, NULL);
    const bool moving = (g_PlayerDpadHeld & 0x0f) != 0;
    if (!stick && tank && moving) {
        const float behind = WrapAngle(PlayerHeading() - s_yaw);
        s_yaw = WrapAngle(s_yaw + behind * 0.08f);
        s_pitch += (kDefaultPitch - s_pitch) * 0.05f;
    }

    // Boom along (yaw, pitch). When the room shell cuts it short, the camera
    // keeps its distance (it may sit behind a wall) and the room renderer
    // clips away everything nearer than that wall.
    const float pitch = s_pitch;
    const float cosP = cosf(pitch), sinP = sinf(pitch);
    const float fwd[3] = { cosP * cosf(s_yaw), sinP, -cosP * sinf(s_yaw) };
    const float h = sqrtf(fwd[0] * fwd[0] + fwd[2] * fwd[2]);
    const float right[3] = { fwd[2] / h, 0.0f, -fwd[0] / h };
    const float pivot[3] = {
        (float)g_playerEntity.position.x + right[0] * kShoulder,
        (float)g_playerEntity.position.y - kEyeHeight,
        (float)g_playerEntity.position.z + right[2] * kShoulder,
    };
    const float back[3] = { -fwd[0], -fwd[1], -fwd[2] };
    const float hit = Room3D_Raycast(pivot, back, kDistance);
    if (s_justEntered) s_distance = kDistance;
    else s_distance += (kDistance - s_distance) * 0.15f;
    // Depth (along the view) of the blocking wall from the camera, plus a
    // margin so the wall itself is cut too.
    s_wallClip = hit < s_distance ? (s_distance - hit) + kWallMargin : 0.0f;
    s_justEntered = false;
    static int s_traceTicks = 0;
    if (s_lastLoggedRoom != (int)g_roomId) { s_traceTicks = 12; s_lastLoggedRoom = g_roomId; }
    if (s_traceTicks > 0) {
        --s_traceTicks;
        dbg_printf("[modern-camera] tick player=(%d,%d,%d) yaw=%.2f pitch=%.2f hit=%.0f dist=%.0f clip=%.0f\n",
                   (int)g_playerEntity.position.x, (int)g_playerEntity.position.y,
                   (int)g_playerEntity.position.z, s_yaw, pitch, hit, s_distance, s_wallClip);
    }

    for (int i = 0; i < 3; ++i) {
        s_from[i] = pivot[i] + back[i] * s_distance;
        s_to[i] = pivot[i] + fwd[i] * 4000.0f;
    }
    s_valid = true;

    // Same record layout as RDT_Camera.cam_from_x .. fov.
    int cam[8] = {
        (int)lroundf(s_from[0]), (int)lroundf(s_from[1]), (int)lroundf(s_from[2]),
        (int)lroundf(s_to[0]), (int)lroundf(s_to[1]), (int)lroundf(s_to[2]),
        0, 0,
    };
    set_scene_render_param(kFocal);
    MatrixToCamera((MATRIX*)cam);
    s_wasActive = true;
}

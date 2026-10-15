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
// Framing follows Resident Evil 5: a nearly level camera at head height just
// behind the right shoulder, the player standing in the left third.
//
// Controls: the right stick orbits (X) and tilts (Y). With tank controls
// (keyboard / D-pad) the camera eases back behind the player while moving.
// Walls pull the camera in, down to a minimum distance that keeps the player
// framed. A wall the camera still ends up behind (right after a door) faces
// away from it and is culled; furniture and door frames in the way are cut
// away by the room renderer's near plane, so the shot always reaches the
// player.
//
// Enabled by [Game] ModernCamera=1 or RE1_MODERN_CAMERA=1.
#include "../Globals.h"
#include "../DebugPrint.h"
#include "../marni/MarniXInput.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

bool  Room3D_MeshAvailable(void);                                       // Room3D.cpp
float Room3D_Raycast(const float origin[3], const float dir[3], float maxDistance,
                     int which);   // 0 all, 1 shell, 2 props                                    // Room3D.cpp

bool g_bModernCamera = false;   // [Game] ModernCamera
char g_testShotName[64] = "";   // set to request one saved frame (MarniDX.cpp)

namespace {

const float kPi = 3.14159265358979f;
const float kEyeHeight = 2700.0f;      // pivot at eye level (RE1 doors ~3300 tall, Chris ~2900; -Y up)
const float kShoulder = 620.0f;        // over-the-right-shoulder offset (player sits left of centre)
const float kDistance = 2400.0f;       // boom length
const float kWallMargin = 220.0f;
const float kMinDistance = 800.0f;     // tight over-the-shoulder; closer would be inside the player
const float kDefaultPitch = 0.12f;     // radians, positive looks down (nearly level, RE5-like)
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
float s_lastShoulder = 0.0f;
int   s_entryTick = 0;
int   s_lastLoggedRoom = -1;

bool Enabled(void)
{
    static const char* value = std::getenv("RE1_MODERN_CAMERA");
    static const bool env = value != NULL && value[0] == '1';
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
    // Player state 8 = the room script is driving the player (cutscenes and
    // scripted walks, e.g. the opening in the main hall before its camera
    // lock is set): keep the authored fixed shots.
    if (g_playerEntity.animationId == 8) return false;
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
    static const bool trace = std::getenv("RE1_MODERN_CAMERA_TRACE") != NULL;
    static int traceTick = 0;
    if (trace && (++traceTick % 10) == 0)
        dbg_printf("[modern-trace] room=%02X cam=%u anim=%u beh=%u msg=%04X lock=%d evt=%08X\n",
                   (unsigned int)g_roomId, (unsigned int)g_roomCameraId,
                   (unsigned int)g_playerEntity.animationId,
                   (unsigned int)g_playerEntity.action_behavior,
                   (unsigned int)g_message_flags,
                   (g_main_state_flags & MSF_CAMERA_LOCK) ? 1 : 0,
                   (unsigned int)g_main_state_flags);
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
        s_entryTick = 0;
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

    // Boom along (yaw, pitch).
    const float head[3] = {
        (float)g_playerEntity.position.x,
        (float)g_playerEntity.position.y - kEyeHeight,
        (float)g_playerEntity.position.z,
    };
    const float pitch = s_pitch;
    const float cosP = cosf(pitch), sinP = sinf(pitch);
    const float viewYaw = s_yaw;
    const float fwd[3] = { cosP * cosf(viewYaw), sinP, -cosP * sinf(viewYaw) };
    const float h = sqrtf(fwd[0] * fwd[0] + fwd[2] * fwd[2]);
    const float right[3] = { fwd[2] / h, 0.0f, -fwd[0] / h };
    // Over-the-shoulder offset, shortened when a wall stands close to the
    // player's right so the pivot never ends up inside or behind it.
    // On a short boom (tight spots) the offset shrinks with it, so the player
    // is not pushed off the left edge of the frame.
    const float shoulderMax = kShoulder * (s_distance < kDistance ? 0.35f + 0.65f * s_distance / kDistance : 1.0f);
    float shoulder = Room3D_Raycast(head, right, kShoulder + kWallMargin, 1) - kWallMargin;
    if (shoulder > shoulderMax) shoulder = shoulderMax;
    if (shoulder < 0.0f) shoulder = 0.0f;
    s_lastShoulder = shoulder;
    const float pivot[3] = {
        head[0] + right[0] * shoulder,
        head[1],
        head[2] + right[2] * shoulder,
    };
    const float back[3] = { -fwd[0], -fwd[1], -fwd[2] };

    // Boom length: walls, floors and ceilings pull the camera in (tight
    // corridors give a close over-the-shoulder view, and the camera stays
    // inside the room so its own walls are never culled away), but never
    // closer than kMinDistance - inside the player.
    const float shellHit = Room3D_Raycast(pivot, back, kDistance + kWallMargin, 1);
    float wanted = shellHit - kWallMargin;
    if (wanted > kDistance) wanted = kDistance;
    if (wanted < kMinDistance) wanted = kMinDistance;
    if (wanted < s_distance || s_justEntered) s_distance = wanted;
    else s_distance += (wanted - s_distance) * 0.12f;
    s_justEntered = false;

    const float camera[3] = {
        pivot[0] + back[0] * s_distance,
        pivot[1] + back[1] * s_distance,
        pivot[2] + back[2] * s_distance,
    };
    // When even the shortest boom is outside the room (a door just walked
    // through), the wall behind faces away from the camera and the room
    // renderer culls it. Furniture and door frames between the camera and
    // the player are cut away by the near plane instead; walls never are, so
    // the room's sides stay whole. Probe from the shoulder pivot, the head
    // and the chest.
    s_wallClip = 0.0f;
    const float targets[3][3] = {
        { pivot[0], pivot[1], pivot[2] },
        { head[0], head[1], head[2] },
        { head[0], head[1] + kEyeHeight * 0.45f, head[2] },
    };
    for (int t = 0; t < 3; ++t) {
        float dir[3] = { camera[0] - targets[t][0], camera[1] - targets[t][1],
                         camera[2] - targets[t][2] };
        const float len = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (len < 1.0f) continue;
        for (int i = 0; i < 3; ++i) dir[i] /= len;
        const float hit = Room3D_Raycast(targets[t], dir, len, 2);
        if (hit < len) {
            // Depth of the obstruction from the camera along the view.
            const float depth = (len - hit) * (fwd[0] * -dir[0] + fwd[1] * -dir[1] + fwd[2] * -dir[2]);
            if (depth + kWallMargin > s_wallClip) s_wallClip = depth + kWallMargin;
        }
    }
    const float hit = shellHit;
    if (s_wallClip > s_distance - 200.0f) s_wallClip = s_distance - 200.0f;   // never past the player
    if (s_wallClip < 0.0f) s_wallClip = 0.0f;

    // Scripted visual checks (RE1_TEST_SHOT_ENTRY=1): ask the renderer for a
    // frame on fixed ticks after each room entry, named after the room.
    static const bool entryShots = std::getenv("RE1_TEST_SHOT_ENTRY") != NULL;
    if (entryShots) {
        static const int kTicks[] = { 1, 4, 10, 25, 50, 90 };
        ++s_entryTick;
        for (size_t k = 0; k < sizeof(kTicks) / sizeof(kTicks[0]); ++k) {
            if (s_entryTick == kTicks[k]) {
                snprintf(g_testShotName, sizeof(g_testShotName), "entry-s%u-r%02X-t%03d.ppm",
                         (unsigned int)g_stageId + 1, (unsigned int)g_roomId, s_entryTick);
            }
        }
    }
    s_justEntered = false;
    static int s_traceTicks = 0;
    if (s_lastLoggedRoom != (int)g_roomId) { s_traceTicks = 12; s_lastLoggedRoom = g_roomId; }
    if (s_traceTicks > 0) {
        --s_traceTicks;
        dbg_printf("[modern-camera] tick player=(%d,%d,%d) yaw=%.2f pitch=%.2f hit=%.0f dist=%.0f clip=%.0f shoulder=%.0f right=(%.2f,%.2f)\n",
                   (int)g_playerEntity.position.x, (int)g_playerEntity.position.y,
                   (int)g_playerEntity.position.z, s_yaw, pitch, hit, s_distance, s_wallClip, s_lastShoulder, right[0], right[2]);
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

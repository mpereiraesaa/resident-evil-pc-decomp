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
const float kSideClearance = 900.0f;  // walls beside the camera fill the frame closer than this
const float kMaxSideShift = 1200.0f;
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
float s_sideShift = 0.0f;      // sideways slide away from walls beside the camera
float s_sideRoom[2] = { 0.0f, 0.0f }, s_sideFree[2] = { 0.0f, 0.0f };   // trace only
bool  s_snapSide = true;
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

// Port addition: evaluate an entry view with the same shoulder and boom
// constraints as the live camera. A floor below the camera is required, but
// a ceiling is not: courtyards and other outdoor rooms also use this camera.
// Room on the tighter side of a camera at `at` looking along `yaw`: the
// least hit * sin(angle) of level rays 25, 45 and 90 degrees off the line of
// sight on either side, against walls and furniture alike (a surface the
// view meets close up at the frame's edge is magnified into a few texels of
// art).
float SideRoom(const float at[3], float yaw)
{
    static const float kAngles[3] = { 0.436f, 0.785f, 1.571f };
    const float flat[3] = { cosf(yaw), 0.0f, -sinf(yaw) };
    const float right[3] = { -sinf(yaw), 0.0f, -cosf(yaw) };
    const float reach = kSideClearance + kMaxSideShift;
    float least = reach;
    for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
        const float sign = sideIndex == 0 ? 1.0f : -1.0f;
        for (int a = 0; a < 3; ++a) {
            const float c = cosf(kAngles[a]), sn = sinf(kAngles[a]);
            const float dir[3] = { flat[0] * c + right[0] * sign * sn, 0.0f,
                                   flat[2] * c + right[2] * sign * sn };
            const float room = Room3D_Raycast(at, dir, reach / sn, 0) * sn;
            if (room < least) least = room;
        }
    }
    return least;
}

float EntryViewClearance(float yaw, float pitch, float* ahead = NULL, float* sideRoom = NULL)
{
    const float head[3] = { (float)g_playerEntity.position.x,
        (float)g_playerEntity.position.y - kEyeHeight,
        (float)g_playerEntity.position.z };
    const float back[3] = { -cosf(pitch) * cosf(yaw), -sinf(pitch),
                            cosf(pitch) * sinf(yaw) };
    const float right[3] = { -sinf(yaw), 0.0f, -cosf(yaw) };
    const float side = Room3D_Raycast(head, right, kShoulder + kWallMargin, 1);
    float distance = kDistance;
    float clearance = 0.0f;
    float camera[3];
    // The shoulder shrinks on a short boom. Resolve that dependency before
    // testing the camera point rather than evaluating an impossible full offset.
    for (int pass = 0; pass < 3; ++pass) {
        const float maximum = kShoulder * (0.35f + 0.65f * distance / kDistance);
        float shoulder = side - kWallMargin;
        if (shoulder < 0.0f) shoulder = 0.0f;
        if (shoulder > maximum) shoulder = maximum;
        const float pivot[3] = { head[0] + right[0] * shoulder, head[1],
                                 head[2] + right[2] * shoulder };
        clearance = Room3D_Raycast(pivot, back, kDistance + kWallMargin, 1) - kWallMargin;
        distance = clearance < kMinDistance ? kMinDistance : clearance;
        if (distance > kDistance) distance = kDistance;
        for (int i = 0; i < 3; ++i) camera[i] = pivot[i] + back[i] * distance;
        if (ahead && pass == 2) {
            const float forward[3] = { -back[0], -back[1], -back[2] };
            *ahead = Room3D_Raycast(pivot, forward, 4000.0f, 1);
        }
        if (sideRoom && pass == 2) *sideRoom = SideRoom(camera, yaw);
    }
    const float down[3] = { 0.0f, 1.0f, 0.0f };
    const float floorLimit = kEyeHeight + 3500.0f;
    if (Room3D_Raycast(camera, down, floorLimit, 1) >= floorLimit) return -1.0f;
    return clearance;
}

// Score both camera space and space ahead, with a penalty for orbiting away
// from the arrival heading. A longer boom facing a nearby wall can be worse
// than a shorter boom looking down the passage the player is entering.
void SelectEntryView(void)
{
    const float heading = s_yaw;
    float ahead = 0.0f;
    EntryViewClearance(heading, kDefaultPitch, &ahead);
    // Keep the arrival view whenever the player looks into open space, even
    // when the shortest boom still sits just behind the door's wall (that
    // wall faces away and is culled, so corridor entries keep looking down
    // the corridor). Search only when the arrival heading faces a wall.
    const bool roomy = ahead >= 2400.0f;
    const float pitches[] = { kDefaultPitch, 0.35f, 0.60f, 0.85f };
    float best = -1.0f;
    float bestScore = -1.0f;
    float bestAhead = 0.0f;
    float bestYaw = heading;
    float bestPitch = kDefaultPitch;
    for (int p = 0; p < (roomy ? 1 : 4); ++p) {
        // Only search steeper views when the level candidates cannot give
        // both a useful boom and space ahead. Avoid extra mesh scans on entry.
        if (p == 1 && best >= 1600.0f && bestAhead >= 2400.0f) break;
        for (int step = 0; step <= (roomy ? 0 : 9); ++step) {
            for (int side = 0; side < (step == 0 || step == 9 ? 1 : 2); ++side) {
                const float offset = (float)step * kPi / 9.0f;
                const float yaw = WrapAngle(heading + (side == 0 ? offset : -offset));
                float room = 0.0f;
                const float clearance = EntryViewClearance(yaw, pitches[p], &ahead, &room);
                if (clearance < kMinDistance) continue;
                // Prefer the level shoulder view; a steeper pitch can fit a
                // doorway without turning all the way around to face it.
                // Room at the frame's sides counts too: a view out of a
                // corner beats one into it.
                const float sideScore = room < kSideClearance ? room : kSideClearance;
                const float score = clearance + ahead * 0.75f + sideScore -
                                    offset * 400.0f - (pitches[p] - kDefaultPitch) * 1000.0f;
                if (score > bestScore) {
                    bestScore = score;
                    best = clearance;
                    bestAhead = ahead;
                    bestYaw = yaw;
                    bestPitch = pitches[p];
                }
            }
        }
    }
    if (best >= kMinDistance) {
        s_yaw = bestYaw;
        s_pitch = bestPitch;
        s_distance = best > kDistance ? kDistance : best;
    }
    dbg_printf("[modern-camera] entry view heading=%.2f yaw=%.2f pitch=%.2f clearance=%.0f supported=%d\n",
               heading, s_yaw, s_pitch, best, best >= kMinDistance ? 1 : 0);
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
        s_sideShift = 0.0f;
        s_snapSide = true;
        s_entryTick = 0;
        s_lastStage = g_stageId;
        s_lastRoom = g_roomId;
        SelectEntryView();
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
        const float yaw = WrapAngle(s_yaw + behind * 0.08f);
        const float pitch = s_pitch + (kDefaultPitch - s_pitch) * 0.05f;
        // Do not ease the chosen entry view back through the arrival wall
        // before the player has actually made enough space for the boom.
        if (EntryViewClearance(yaw, pitch) >= kMinDistance) {
            s_yaw = yaw;
            s_pitch = pitch;
        }
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

    // Side clearance: a wall beside the camera, or one the view meets at an
    // angle just ahead of it, fills that side of the frame with a few
    // magnified texels of art. Rays across the view (25, 45 and 90 degrees
    // off the line of sight, each side) measure the room on each side as
    // hit * sin(angle). The camera slides sideways by the least amount that
    // clears both sides, or to the middle when the gap is too narrow; if even
    // the middle is cramped, the boom is pulled in to where the space opens
    // up around the player.
    const float flat[3] = { fwd[0] / h, 0.0f, fwd[2] / h };
    const auto measure = [&](const float at[3], float room[2], float free[2]) {
        static const float kAngles[3] = { 0.436f, 0.785f, 1.571f };
        const float reach = kSideClearance + kMaxSideShift;
        for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
            const float sign = sideIndex == 0 ? 1.0f : -1.0f;
            room[sideIndex] = free[sideIndex] = reach;
            for (int a = 0; a < 3; ++a) {
                const float c = cosf(kAngles[a]), sn = sinf(kAngles[a]);
                const float dir[3] = { flat[0] * c + right[0] * sign * sn, 0.0f,
                                       flat[2] * c + right[2] * sign * sn };
                const float hit = Room3D_Raycast(at, dir, reach / sn, 0);
                if (hit * sn < room[sideIndex]) room[sideIndex] = hit * sn;
                if (a == 2) free[sideIndex] = hit;
            }
        }
    };
    // Slide (+ right) for measured room; *cleared = the room left on the
    // tighter side afterwards.
    const auto solve = [&](const float room[2], const float free[2], float* cleared) {
        float target;
        if (room[0] + room[1] >= 2.0f * kSideClearance) {
            const float low = kSideClearance - room[1], high = room[0] - kSideClearance;
            target = 0.0f < low ? low : (0.0f > high ? high : 0.0f);
        } else {
            target = (room[0] - room[1]) * 0.5f;
        }
        if (target > free[0] - kWallMargin) target = free[0] - kWallMargin;
        if (target < -(free[1] - kWallMargin)) target = -(free[1] - kWallMargin);
        if (target > kMaxSideShift) target = kMaxSideShift;
        if (target < -kMaxSideShift) target = -kMaxSideShift;
        const float r = room[0] - target, l = room[1] + target;
        *cleared = r < l ? r : l;
        return target;
    };
    // Evaluated at the boom's target length (not its eased length), so the
    // choice is stable while the player stands still.
    float sideRoom[2], sideFree[2], cleared = 0.0f;
    float onBoom[3];
    for (int i = 0; i < 3; ++i) onBoom[i] = pivot[i] + back[i] * wanted;
    measure(onBoom, sideRoom, sideFree);
    float target = solve(sideRoom, sideFree, &cleared);
    if (cleared < kSideClearance && wanted > kMinDistance) {
        float bestDistance = wanted, bestCleared = cleared, bestTarget = target;
        static const float kFractions[3] = { 0.75f, 0.5f, 0.0f };
        for (int k = 0; k < 3 && bestCleared < kSideClearance; ++k) {
            float d = wanted * kFractions[k];
            if (d < kMinDistance) d = kMinDistance;
            if (d >= bestDistance) continue;
            float at[3], room[2], free[2], c;
            for (int i = 0; i < 3; ++i) at[i] = pivot[i] + back[i] * d;
            measure(at, room, free);
            const float t = solve(room, free, &c);
            if (c > bestCleared + 50.0f) {
                bestDistance = d; bestCleared = c; bestTarget = t;
                sideRoom[0] = room[0]; sideRoom[1] = room[1];
            }
        }
        wanted = bestDistance;
        target = bestTarget;
    }
    if (wanted < s_distance || s_justEntered) s_distance = wanted;
    else s_distance += (wanted - s_distance) * 0.12f;
    s_justEntered = false;
    // The slide is clamped against the walls where the camera actually is.
    for (int i = 0; i < 3; ++i) onBoom[i] = pivot[i] + back[i] * s_distance;
    for (int sideIndex = 0; sideIndex < 2; ++sideIndex) {
        const float sign = sideIndex == 0 ? 1.0f : -1.0f;
        const float dir[3] = { right[0] * sign, 0.0f, right[2] * sign };
        sideFree[sideIndex] = Room3D_Raycast(onBoom, dir, kSideClearance + kMaxSideShift, 1);
    }
    for (int k = 0; k < 2; ++k) { s_sideRoom[k] = sideRoom[k]; s_sideFree[k] = sideFree[k]; }
    if (s_snapSide) s_sideShift = target;
    else s_sideShift += (target - s_sideShift) * 0.15f;
    s_snapSide = false;
    // Easing can lag behind a wall that is getting closer: clamp to the space
    // actually available this tick.
    if (s_sideShift > sideFree[0] - kWallMargin) s_sideShift = sideFree[0] - kWallMargin;
    if (s_sideShift < -(sideFree[1] - kWallMargin)) s_sideShift = -(sideFree[1] - kWallMargin);
    const float camera[3] = {
        onBoom[0] + right[0] * s_sideShift,
        onBoom[1],
        onBoom[2] + right[2] * s_sideShift,
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
        dbg_printf("[modern-camera] tick player=(%d,%d,%d) yaw=%.2f pitch=%.2f hit=%.0f dist=%.0f clip=%.0f shoulder=%.0f side=%.0f room=%.0f/%.0f free=%.0f/%.0f right=(%.2f,%.2f)\n",
                   (int)g_playerEntity.position.x, (int)g_playerEntity.position.y,
                   (int)g_playerEntity.position.z, s_yaw, pitch, hit, s_distance, s_wallClip, s_lastShoulder, s_sideShift, s_sideRoom[0], s_sideRoom[1], s_sideFree[0], s_sideFree[1], right[0], right[2]);
    }

    for (int i = 0; i < 3; ++i) {
        s_from[i] = camera[i];
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

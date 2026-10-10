#include "TranslationBank.h"
#include "../DebugPrint.h"
#include "../platform/platform.h"
#include "../system/AssetPath.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct TranslationBank {
    unsigned char* data;
    size_t size;
    unsigned int count;
    int attempted;
};

static TranslationBank s_global = {};
static TranslationBank s_descriptions = {};
static TranslationBank s_itemNames = {};
static TranslationBank s_room = {};
static char s_roomPath[128] = "";

static unsigned int read_u32(const unsigned char* data)
{
    return (unsigned int)data[0] |
           ((unsigned int)data[1] << 8) |
           ((unsigned int)data[2] << 16) |
           ((unsigned int)data[3] << 24);
}

static void clear_bank(TranslationBank* bank)
{
    free(bank->data);
    memset(bank, 0, sizeof(*bank));
}

static int load_bank(TranslationBank* bank, const char* path)
{
    if (bank->attempted) return bank->data != NULL;
    bank->attempted = 1;

    char resolved[520];
    char normalized[520];
    const char* asset = ResolveAssetRoot(path, resolved, sizeof(resolved));
    const char* nativePath = plat_normalize_path(asset, normalized,
                                                 sizeof(normalized));
    FILE* file = fopen(nativePath, "rb");
    if (file == NULL) return 0;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }
    long fileSize = ftell(file);
    if (fileSize < 16 || fileSize > 16 * 1024 * 1024 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }

    unsigned char* data = (unsigned char*)malloc((size_t)fileSize);
    if (data == NULL || fread(data, 1, (size_t)fileSize, file) != (size_t)fileSize) {
        free(data);
        fclose(file);
        return 0;
    }
    fclose(file);

    static const unsigned char magic[8] = {'R','E','1','T','X','T','1',0};
    unsigned int count = read_u32(data + 8);
    size_t tableEnd = 12 + ((size_t)count + 1) * 4;
    if (memcmp(data, magic, sizeof(magic)) != 0 || count > 4096 ||
        tableEnd > (size_t)fileSize) {
        free(data);
        return 0;
    }

    unsigned int previous = read_u32(data + 12);
    if (previous != tableEnd) {
        free(data);
        return 0;
    }
    for (unsigned int i = 1; i <= count; ++i) {
        unsigned int next = read_u32(data + 12 + i * 4);
        if (next < previous || next > (unsigned int)fileSize) {
            free(data);
            return 0;
        }
        previous = next;
    }
    if (previous != (unsigned int)fileSize) {
        free(data);
        return 0;
    }

    bank->data = data;
    bank->size = (size_t)fileSize;
    bank->count = count;
    dbg_printf("[text] translation bank active: %s (%u entries)\n",
               nativePath, count);
    return 1;
}

static const unsigned char* bank_entry(TranslationBank* bank,
                                       const char* path,
                                       unsigned int index)
{
    if (!load_bank(bank, path) || index >= bank->count) return NULL;
    unsigned int offset = read_u32(bank->data + 12 + index * 4);
    unsigned int end = read_u32(bank->data + 12 + (index + 1) * 4);
    if (offset >= end || end > bank->size) return NULL;
    return bank->data + offset;
}

const unsigned char* Translation_GlobalMessage(unsigned int index)
{
    if (GetAssetVersion() != 0) return NULL;
    return bank_entry(&s_global,
                      GAME_DATA_ROOT "translation\\global.bin", index);
}

const unsigned char* Translation_ItemDescription(unsigned int index)
{
    if (GetAssetVersion() != 0) return NULL;
    return bank_entry(&s_descriptions,
                      GAME_DATA_ROOT "translation\\descriptions.bin", index);
}

const unsigned char* Translation_ItemName(unsigned int index)
{
    if (GetAssetVersion() != 0) return NULL;
    return bank_entry(&s_itemNames,
                      GAME_DATA_ROOT "translation\\item_names.bin", index);
}

const unsigned char* Translation_RoomMessage(unsigned char stage,
                                             unsigned char room,
                                             unsigned char variant,
                                             unsigned int index)
{
    if (GetAssetVersion() != 0) return NULL;

    char path[128];
    sprintf_s(path, sizeof(path),
              GAME_DATA_ROOT "translation\\room\\room_%x%02x_%u.bin",
              (unsigned int)stage + 1, (unsigned int)room,
              (unsigned int)variant);
    if (strcmp(path, s_roomPath) != 0) {
        clear_bank(&s_room);
        strcpy_s(s_roomPath, sizeof(s_roomPath), path);
    }
    return bank_entry(&s_room, s_roomPath, index);
}

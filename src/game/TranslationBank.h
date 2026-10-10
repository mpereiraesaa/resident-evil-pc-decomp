#pragma once

// Optional native text overrides. Every function returns NULL when the
// selected mod does not provide a compatible bank or the index is absent.
const unsigned char* Translation_GlobalMessage(unsigned int index);
const unsigned char* Translation_ItemDescription(unsigned int index);
const unsigned char* Translation_ItemName(unsigned int index);
const unsigned char* Translation_RoomMessage(unsigned char stage,
                                             unsigned char room,
                                             unsigned char variant,
                                             unsigned int index);

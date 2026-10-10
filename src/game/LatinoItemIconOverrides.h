#pragma once

// The Latino HD pack was authored against a different PC release, so its
// Classic REbirth CRC names do not match the GOG-USA ITEM_ALL.PIX sprites.
// Map the game's stable image-type index to the corresponding CRC-named WebP.
// Missing files and null entries fall back to the game's indexed sprite (or
// the normal CRC lookup used by other texture pages).
static const char* const kLatinoItemIconCrc[72] = {
    "7144F156", "CB9051FA", "767A0415", "000E9601", "7B8F6B96", "EAD4FED9",
    "D719C16E", "8A02A5CB", "CBD5F105", "9439EC5C", "AA9AAA44", "DF666340",
    nullptr,    "7E82650B", "964E5E83", "450B7BD9", "C25A515D", "40E6F2A1",
    "6235F5EB", "9E61E10A", "76CA10FE", "366A7E25", "7BF24854", "62DEA61A",
    "4ADD3BA5", "2E827BDC", "2A2D2E44", "00B1D431", "7051C288", "C7B5A02E",
    "0D73EBF7", nullptr,    "DC4B25E6", "C4295579", nullptr,    nullptr,
    "507A6451", "51918184", "25BBC411", "CB1782F6", "A1A75C5A", "7AD58279",
    "C9B490B0", "2344BAC1", "FD6B6690", "125BFAF2", nullptr,    "517B963E",
    "AB77217F", "4F8DC05B", "7FF9E89F", "4B2E0C17", "DA691AA8", "F91887B9",
    "0994766F", "FBAE873D", "6392875E", "B0A0622A", "D856F228", nullptr,
    "7C80B68B", "523C4181", "F8F65525", "ED7F38AA", "9596F0F6", "B377F532",
    "136BC178", "D4F69FCF", "B29AF176", "BFB626ED", "422D675E", "E9F2A5D6",
};

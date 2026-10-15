// MarniDX.cpp - D3D11 backend implementation for the Marni System shim.
//
// This file is the ONLY translation unit in the project that owns D3D11,
// DXGI, and the HLSL/D3DCompiler pipeline. Everything exposed to the game
// layer goes through the MarniDX class declared in MarniDX.h, using opaque
// MarniHandle texture handles. No <d3d11.h>-derived type is visible outside
// this file (verified: none of the game-layer headers include d3d11.h).
//
// The body is a faithful re-home of the pipeline that used to live mixed into
// MarniSystem.cpp / MarniBits.cpp / CharacterSelectionScreen.cpp, with the
// raw COM pointers moved into a single private Impl. Behaviour is unchanged
// from the previous working build — just re-encapsulated.
#include "MarniDX.h"

#include <new>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include "../game/Types.h"
#include "../third_party/smaa/AreaTex.h"
#include "../third_party/smaa/SearchTex.h"

// config.ini [Display] VSync (Globals.cpp). Declared locally rather than
// pulling all of Globals.h into the Marni layer.
extern BOOL g_bVSync;
extern bool g_bSmaa;
extern bool g_bFrameInterpolation;

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

// Forward-declare the game-layer structs we only pass by opaque pointer.
struct DisplayModeInfo;
struct D3DRendererInfo;

// ============================================================================
// Shader sources (compiled at runtime via D3DCompile).
// ============================================================================
static const char* g_QuadVS_Source = R"(
cbuffer SpriteCB : register(b0) { row_major float4x4 g_MVP; };
struct VS_INPUT  { float2 pos:POSITION; float2 tex:TEXCOORD0; float4 col:COLOR0; };
struct VS_OUTPUT { float4 pos:SV_Position; float2 tex:TEXCOORD0; float4 col:COLOR0; float depth:TEXCOORD1; };
VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT o;
    o.pos = mul(float4(input.pos.x, input.pos.y, 0.0f, 1.0f), g_MVP);
    o.tex = input.tex;
    o.col = input.col;
    o.depth = 0.0f;
    return o;
}
)";

// 3D model VS: same screen-space ortho for x/y, but the vertex carries a
// pre-normalised [0,1] linear depth. NDC z provides clip coordinates; the
// pixel shader writes the perspective-correct depth used by the depth buffer. The original leaned on a real Z-buffer for this
// ("MarniSystem Direct3D::MD3DCreateZBuffer"); a per-triangle painter sort
// cannot, and made faces pop in and out as a model turned.
//
// pos.w carries the vertex's VIEW-SPACE Z. The caller already did the divide
// when it projected to screen space, so emitting SV_Position with w = 1 made
// the rasteriser interpolate UV and colour LINEARLY IN SCREEN SPACE - affine
// mapping, i.e. the PS1 texture swim. Pre-multiplying x/y/z by w and handing
// the rasteriser that w restores the same screen position after the perspective
// divide while making every interpolator perspective-correct. It shows up
// worst on big polygons close to the camera at an oblique angle - the door
// panel in the room-transition animation.
static const char* g_Model3DVS_Source = R"(
cbuffer SpriteCB : register(b0) { row_major float4x4 g_MVP; };
struct VS_INPUT  { float4 pos:POSITION; float2 tex:TEXCOORD0; float4 col:COLOR0; };
struct VS_OUTPUT { float4 pos:SV_Position; float2 tex:TEXCOORD0; float4 col:COLOR0; float depth:TEXCOORD1; };
VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT o;
    float4 p = mul(float4(input.pos.x, input.pos.y, 0.0f, 1.0f), g_MVP);
    float w = max(input.pos.w, 1e-4f);
    o.pos = float4(p.x * w, p.y * w, input.pos.z * w, w);
    o.tex = input.tex;
    o.col = input.col;
    o.depth = input.pos.z;
    return o;
}
)";

// Callers supply LINEAR view-depth. Hardware SV_Position.z interpolation is
// affine in screen space, so it cannot interpolate that depth across an oblique
// polygon. Carry it as a perspective-correct varying and write pixel depth.
// This keeps models, room meshes and shadows on their shared linear ramp.
static const char* g_QuadPS_Source = R"(
Texture2D    g_Texture : register(t0);
SamplerState g_Sampler  : register(s0);
struct PS_INPUT { float4 pos:SV_Position; float2 tex:TEXCOORD0; float4 col:COLOR0; float depth:TEXCOORD1; };
struct PS_OUTPUT { float4 color:SV_Target; float depth:SV_Depth; };
PS_OUTPUT main(PS_INPUT i)
{
    PS_OUTPUT o;
    o.color = g_Texture.Sample(g_Sampler, i.tex) * i.col;
    o.depth = i.depth;
    return o;
}
)";

// ============================================================================
// Internal types (file-local, never exported)
// ============================================================================
struct QuadVertex {
    float x, y;         // screen position
    float u, v;         // texture coords
    float r, g, b, a;   // tint color 0..1
};

// Depth-buffered variant used by the TMD model path.
struct Model3DVertex {
    float x, y, z;      // screen position + normalised [0,1] depth
    float w;            // view-space Z; the VS uses it to restore perspective-
                        // correct interpolation of u/v and the vertex colour
    float u, v;
    float r, g, b, a;
};

struct SpriteConstantBuffer {
    float mvp[4][4];
};

// Handle-table slot: maps a public MarniHandle (1-based index) to its D3D11
// objects. Slot 0 is reserved for MARNI_NULL_HANDLE.
struct TexSlot {
    ID3D11Texture2D*        tex;
    ID3D11ShaderResourceView* srv;
    int                     width;
    int                     height;
};

#define MARNI_MAX_TEXTURES 2048

// ============================================================================
// MarniDX::Impl - all D3D11 ownership lives here.
// ============================================================================
struct MarniDX::Impl {
    // core device
    ID3D11Device*           device    = nullptr;
    ID3D11DeviceContext*    context   = nullptr;
    IDXGISwapChain*          swapChain = nullptr;
    HWND                     hWnd      = nullptr;

    // render target
    ID3D11RenderTargetView*  rtv       = nullptr;
    ID3D11Texture2D*         depthStencil    = nullptr;
    ID3D11DepthStencilView*   depthStencilView = nullptr;

    // Optional SMAA 1x post-process chain. Scene color is rendered offscreen;
    // edge and blend weights are intermediate single-channel render targets.
    ID3D11Texture2D*         smaaScene = nullptr;
    ID3D11RenderTargetView*  smaaSceneRTV = nullptr;
    ID3D11ShaderResourceView* smaaSceneSRV = nullptr;
    ID3D11Texture2D*         smaaEdges = nullptr;
    ID3D11RenderTargetView*  smaaEdgesRTV = nullptr;
    ID3D11ShaderResourceView* smaaEdgesSRV = nullptr;
    ID3D11Texture2D*         smaaWeights = nullptr;
    ID3D11RenderTargetView*  smaaWeightsRTV = nullptr;
    ID3D11ShaderResourceView* smaaWeightsSRV = nullptr;
    ID3D11ShaderResourceView* smaaAreaSRV = nullptr;
    ID3D11ShaderResourceView* smaaSearchSRV = nullptr;
    ID3D11VertexShader*       smaaVS = nullptr;
    ID3D11PixelShader*        smaaEdgePS = nullptr;
    ID3D11PixelShader*        smaaWeightPS = nullptr;
    ID3D11PixelShader*        smaaBlendPS = nullptr;
    ID3D11Buffer*             smaaCB = nullptr;
    bool                      smaaEnabled = false;

    ID3D11Texture2D*          frameCurrent = nullptr;
    ID3D11ShaderResourceView* frameCurrentSRV = nullptr;
    bool                       frameInterpolation = false;

    // pipeline state
    ID3D11RasterizerState*   rasterScissor = nullptr;
    ID3D11BlendState*        blendAlpha    = nullptr;
    ID3D11BlendState*        blendAdd      = nullptr;
    ID3D11BlendState*        blendDisabled = nullptr;
    ID3D11SamplerState*      sampLinear    = nullptr;
    ID3D11SamplerState*      sampPoint     = nullptr;
    ID3D11DepthStencilState* depthDisabled = nullptr;
    ID3D11DepthStencilState* depthEnabled  = nullptr;
    // Test-only depth (write off) for the translucent ground-shadow / blood
    // pool quads: the original rendered them as world-space viewport quads
    // against the same Z-buffer as the TMD objects, so a character in front
    // clipped them no matter where the ordering table put the primitive.
    ID3D11DepthStencilState* depthTestNoWrite = nullptr;

    // shaders / buffers
    ID3D11VertexShader*      quadVS        = nullptr;
    ID3D11PixelShader*       quadPS        = nullptr;
    ID3D11InputLayout*       quadLayout    = nullptr;
    ID3D11Buffer*            quadVB        = nullptr;
    ID3D11Buffer*            spriteCB      = nullptr;

    // depth-buffered 3D model pipeline (shares quadPS and spriteCB)
    ID3D11VertexShader*      model3DVS     = nullptr;
    ID3D11InputLayout*       model3DLayout = nullptr;
    ID3D11Buffer*            model3DVB     = nullptr;

    // fallback white 1x1 texture + SRV (handle index 1 reserved)
    ID3D11Texture2D*         whiteTex      = nullptr;
    ID3D11ShaderResourceView* whiteSRV    = nullptr;

    // back buffer dimensions (clamped to >= 320x240)
    DWORD                    width         = 0;
    DWORD                    height        = 0;
    BOOL                     ready         = FALSE;
    MarniPresentationMode    presentation  = MARNI_PRESENT_STRETCH;
    float                    verticalPan   = 0.0f;

    // texture handle table (index 0 reserved = NULL)
    TexSlot                  slots[MARNI_MAX_TEXTURES];

    // font cache (handle/dims only — D3D11 pointer stays in slots[])
    MarniHandle              fontHandle    = MARNI_NULL_HANDLE;
    int                      fontW         = 0;
    int                      fontH         = 0;

    //--- helpers ---------------------------------------------------------
    void ReleaseSlot(int idx);
    int  AllocSlot(ID3D11Texture2D* t, ID3D11ShaderResourceView* s,
                   int w, int h);
    void ReleaseAllState();
    void ReleaseAllTextures();
    void ReleaseSmaa();
    BOOL CreateSmaa();
    BOOL ResizeSmaa();
    void RenderSmaa();
    void ReleaseFrameInterpolation();
    BOOL CreateFrameInterpolation();
    BOOL ResizeFrameInterpolation();
};

// ----------------------------------------------------------------------------
// Global singleton
// ----------------------------------------------------------------------------
static MarniDX* g_pDX = nullptr;

MarniDX* Marni_DX() { return g_pDX; }

MarniDX* MarniDX_Create()
{
    if (g_pDX) return g_pDX;
    g_pDX = new (std::nothrow) MarniDX();
    return g_pDX;
}

void MarniDX_DestroyGlobal()
{
    if (g_pDX) {
        g_pDX->Destroy();
        delete g_pDX;
        g_pDX = nullptr;
    }
}

MarniDX::MarniDX()  : m_pImpl(new MarniDX::Impl()) {}
MarniDX::~MarniDX() { delete m_pImpl; m_pImpl = nullptr; }

// ============================================================================
// Local helpers (file-local, all D3D11 stays in this TU)
// ============================================================================
static void BuildOrthoMatrix(float* m, float left, float right,
                             float bottom, float top)
{
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0]  = 2.0f / (right - left);
    m[5]  = 2.0f / (top - bottom);
    m[10] = 1.0f;
    m[12] = (left + right) / (left - right);
    m[13] = (top + bottom) / (bottom - top);
    m[15] = 1.0f;
}

static D3D11_RECT BuildPresentationScissor(MarniDX::Impl* p)
{
    D3D11_RECT rect = { 0, 0,
        (LONG)(p && p->width ? p->width : 640),
        (LONG)(p && p->height ? p->height : 480) };
    if (!p || !p->width || !p->height ||
        p->presentation == MARNI_PRESENT_STRETCH)
        return rect;

    const float outputW = (float)p->width;
    const float outputH = (float)p->height;
    float aspect = outputW / outputH;
    if (aspect < (4.0f / 3.0f)) aspect = 4.0f / 3.0f;
    if (aspect > (16.0f / 9.0f)) aspect = 16.0f / 9.0f;

    float viewportW = outputW;
    float viewportH = viewportW / aspect;
    if (viewportH > outputH) {
        viewportH = outputH;
        viewportW = viewportH * aspect;
    }
    const float viewportX = (outputW - viewportW) * 0.5f;
    const float viewportY = (outputH - viewportH) * 0.5f;
    float x = viewportX, y = viewportY, w = viewportW, h = viewportH;

    if (p->presentation == MARNI_PRESENT_FIT_4_3) {
        w = viewportW;
        h = w * (3.0f / 4.0f);
        if (h > viewportH) {
            h = viewportH;
            w = h * (4.0f / 3.0f);
        }
        x = viewportX + (viewportW - w) * 0.5f;
        y = viewportY + (viewportH - h) * 0.5f;
    }

    rect.left   = (LONG)floorf(x);
    rect.top    = (LONG)floorf(y);
    rect.right  = (LONG)ceilf(x + w);
    rect.bottom = (LONG)ceilf(y + h);
    return rect;
}

static void BuildPresentationMatrix(float* m, MarniDX::Impl* p)
{
    if (!p || p->width == 0 || p->height == 0 ||
        p->presentation == MARNI_PRESENT_STRETCH) {
        const float width = p && p->width ? (float)p->width : 640.0f;
        const float height = p && p->height ? (float)p->height : 480.0f;
        BuildOrthoMatrix(m, 0.0f, width, height, 0.0f);
        return;
    }

    const float outputW = (float)p->width;
    const float outputH = (float)p->height;
    float aspect = outputW / outputH;
    if (aspect < (4.0f / 3.0f)) aspect = 4.0f / 3.0f;
    if (aspect > (16.0f / 9.0f)) aspect = 16.0f / 9.0f;

    float viewportW = outputW;
    float viewportH = viewportW / aspect;
    if (viewportH > outputH) {
        viewportH = outputH;
        viewportW = viewportH * aspect;
    }
    const float viewportX = (outputW - viewportW) * 0.5f;
    const float viewportY = (outputH - viewportH) * 0.5f;

    float scaleX, scaleY, offsetX, offsetY;
    if (p->presentation == MARNI_PRESENT_CROP_16_9) {
        const float uniformScale = viewportW / 320.0f;
        const float visibleHeight = fminf(240.0f, 320.0f * viewportH / viewportW);
        const float maxPan = 240.0f - visibleHeight;
        float pan = p->verticalPan;
        if (pan < 0.0f) pan = 0.0f;
        if (pan > maxPan) pan = maxPan;
        scaleX = viewportW / outputW;
        scaleY = uniformScale * 240.0f / outputH;
        offsetX = viewportX;
        offsetY = viewportY - pan * uniformScale;
    } else {
        float imageW = viewportW;
        float imageH = imageW * (3.0f / 4.0f);
        if (imageH > viewportH) {
            imageH = viewportH;
            imageW = imageH * (4.0f / 3.0f);
        }
        scaleX = imageW / outputW;
        scaleY = imageH / outputH;
        offsetX = viewportX + (viewportW - imageW) * 0.5f;
        offsetY = viewportY + (viewportH - imageH) * 0.5f;
    }

    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = 2.0f * scaleX / outputW;
    m[5] = -2.0f * scaleY / outputH;
    m[10] = 1.0f;
    m[12] = 2.0f * offsetX / outputW - 1.0f;
    m[13] = 1.0f - 2.0f * offsetY / outputH;
    m[15] = 1.0f;
}

static bool CompileShaders(ID3D11Device* dev,
                           ID3D11VertexShader** outVS,
                           ID3D11PixelShader**  outPS,
                           ID3D11InputLayout**  outLayout)
{
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* errBlob = nullptr;

    HRESULT hr = D3DCompile(g_QuadVS_Source, strlen(g_QuadVS_Source),
        "QuadVS", nullptr, nullptr, "main", "vs_4_0",
        D3DCOMPILE_ENABLE_STRICTNESS, 0, &vsBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { OutputDebugStringA("[MarniDX] VS compile: ");
            OutputDebugStringA((char*)errBlob->GetBufferPointer());
            OutputDebugStringA("\n"); errBlob->Release(); }
        return false;
    }

    hr = D3DCompile(g_QuadPS_Source, strlen(g_QuadPS_Source),
        "QuadPS", nullptr, nullptr, "main", "ps_4_0",
        D3DCOMPILE_ENABLE_STRICTNESS, 0, &psBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { OutputDebugStringA("[MarniDX] PS compile: ");
            OutputDebugStringA((char*)errBlob->GetBufferPointer());
            OutputDebugStringA("\n"); errBlob->Release(); }
        vsBlob->Release();
        return false;
    }

    hr = dev->CreateVertexShader(vsBlob->GetBufferPointer(),
        vsBlob->GetBufferSize(), nullptr, outVS);
    if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }

    hr = dev->CreatePixelShader(psBlob->GetBufferPointer(),
        psBlob->GetBufferSize(), nullptr, outPS);
    if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 8,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = dev->CreateInputLayout(layout, 3, vsBlob->GetBufferPointer(),
        vsBlob->GetBufferSize(), outLayout);
    vsBlob->Release(); psBlob->Release();
    return SUCCEEDED(hr);
}

// Compile the depth-buffered model VS + its input layout. Reuses the quad PS,
// whose PS_INPUT signature is identical.
static bool CompileModel3DShader(ID3D11Device* dev,
                                 ID3D11VertexShader** outVS,
                                 ID3D11InputLayout**  outLayout)
{
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* errBlob = nullptr;

    HRESULT hr = D3DCompile(g_Model3DVS_Source, strlen(g_Model3DVS_Source),
        "Model3DVS", nullptr, nullptr, "main", "vs_4_0",
        D3DCOMPILE_ENABLE_STRICTNESS, 0, &vsBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { OutputDebugStringA("[MarniDX] model VS compile: ");
            OutputDebugStringA((char*)errBlob->GetBufferPointer());
            OutputDebugStringA("\n"); errBlob->Release(); }
        return false;
    }

    hr = dev->CreateVertexShader(vsBlob->GetBufferPointer(),
        vsBlob->GetBufferSize(), nullptr, outVS);
    if (FAILED(hr)) { vsBlob->Release(); return false; }

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = dev->CreateInputLayout(layout, 3, vsBlob->GetBufferPointer(),
        vsBlob->GetBufferSize(), outLayout);
    vsBlob->Release();
    return SUCCEEDED(hr);
}

// SMAA 1x uses the official three-pass HLSL implementation. The shader stays
// as a shipped source file so upstream license/provenance remain auditable.
static const char* g_SmaaWrappers = R"(
Texture2D ColorTex : register(t0);
Texture2D EdgesTex : register(t1);
Texture2D AreaTex : register(t2);
Texture2D SearchTex : register(t3);
Texture2D WeightsTex : register(t4);
struct SmaaIn { float4 pos:SV_Position; float2 uv:TEXCOORD0; float4 o0:TEXCOORD1; float4 o1:TEXCOORD2; float4 o2:TEXCOORD3; float2 pix:TEXCOORD4; };
SmaaIn SmaaVS(uint id:SV_VertexID) {
    SmaaIn o; float2 uv=float2((id<<1)&2, id&2);
    o.pos=float4(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0); o.uv=uv;
    float4 e[3]; SMAAEdgeDetectionVS(uv,e); o.o0=e[0]; o.o1=e[1]; o.o2=e[2];
    float2 p; float4 b[3]; SMAABlendingWeightCalculationVS(uv,p,b); o.pix=p;
    return o;
}
float4 SmaaEdgePS(SmaaIn i):SV_Target { float4 o[3]={i.o0,i.o1,i.o2}; return float4(SMAALumaEdgeDetectionPS(i.uv,o,ColorTex),0,1); }
float4 SmaaWeightPS(SmaaIn i):SV_Target { float4 o[3]={i.o0,i.o1,i.o2}; return SMAABlendingWeightCalculationPS(i.uv,i.pix,o,EdgesTex,AreaTex,SearchTex,float4(0,0,0,0)); }
float4 SmaaBlendPS(SmaaIn i):SV_Target { float4 o; SMAANeighborhoodBlendingVS(i.uv,o); return SMAANeighborhoodBlendingPS(i.uv,o,ColorTex,WeightsTex); }
)";

static bool MakeFrameHistoryTexture(ID3D11Device* device, DWORD width,
                                    DWORD height, ID3D11Texture2D** tex,
                                    ID3D11ShaderResourceView** srv)
{
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = width; td.Height = height; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&td, nullptr, tex))) return false;
    if (FAILED(device->CreateShaderResourceView(*tex, nullptr, srv))) return false;
    return true;
}

static bool CompileSmaaShader(const std::string& source, const char* entry,
                              const char* profile, ID3DBlob** out)
{
    ID3DBlob* errors = nullptr;
    HRESULT hr = D3DCompile(source.data(), source.size(), "SMAA.hlsl", nullptr,
        nullptr, entry, profile, D3DCOMPILE_ENABLE_STRICTNESS, 0, out, &errors);
    if (FAILED(hr)) {
        OutputDebugStringA("[SMAA] shader compile failed: ");
        if (errors) OutputDebugStringA((const char*)errors->GetBufferPointer());
        OutputDebugStringA("\n");
    }
    if (errors) errors->Release();
    return SUCCEEDED(hr);
}

static bool MakeSmaaTarget(ID3D11Device* device, DWORD width, DWORD height,
                           DXGI_FORMAT format, ID3D11Texture2D** tex,
                           ID3D11RenderTargetView** rtv,
                           ID3D11ShaderResourceView** srv)
{
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = width; td.Height = height; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = format; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&td, nullptr, tex))) return false;
    if (FAILED(device->CreateRenderTargetView(*tex, nullptr, rtv)) ||
        FAILED(device->CreateShaderResourceView(*tex, nullptr, srv))) return false;
    return true;
}

void MarniDX::Impl::ReleaseFrameInterpolation()
{
    frameInterpolation = false;
    if (frameCurrentSRV) { frameCurrentSRV->Release(); frameCurrentSRV = nullptr; }
    if (frameCurrent) { frameCurrent->Release(); frameCurrent = nullptr; }
}

BOOL MarniDX::Impl::ResizeFrameInterpolation()
{
    if (frameCurrentSRV) { frameCurrentSRV->Release(); frameCurrentSRV = nullptr; }
    if (frameCurrent) { frameCurrent->Release(); frameCurrent = nullptr; }
    return MakeFrameHistoryTexture(device, width, height, &frameCurrent,
                                   &frameCurrentSRV);
}

BOOL MarniDX::Impl::CreateFrameInterpolation()
{
    if (!g_bFrameInterpolation) return TRUE;
    if (!ResizeFrameInterpolation()) { ReleaseFrameInterpolation(); return FALSE; }
    frameInterpolation = true;
    OutputDebugStringA("[FrameInterpolation] 30 Hz simulation / 60 Hz pose rendering enabled\n");
    return TRUE;
}

void MarniDX::Impl::ReleaseSmaa()
{
    if (smaaSceneSRV) { smaaSceneSRV->Release(); smaaSceneSRV = nullptr; }
    if (smaaSceneRTV) { smaaSceneRTV->Release(); smaaSceneRTV = nullptr; }
    if (smaaScene) { smaaScene->Release(); smaaScene = nullptr; }
    if (smaaEdgesSRV) { smaaEdgesSRV->Release(); smaaEdgesSRV = nullptr; }
    if (smaaEdgesRTV) { smaaEdgesRTV->Release(); smaaEdgesRTV = nullptr; }
    if (smaaEdges) { smaaEdges->Release(); smaaEdges = nullptr; }
    if (smaaWeightsSRV) { smaaWeightsSRV->Release(); smaaWeightsSRV = nullptr; }
    if (smaaWeightsRTV) { smaaWeightsRTV->Release(); smaaWeightsRTV = nullptr; }
    if (smaaWeights) { smaaWeights->Release(); smaaWeights = nullptr; }
    if (smaaAreaSRV) { smaaAreaSRV->Release(); smaaAreaSRV = nullptr; }
    if (smaaSearchSRV) { smaaSearchSRV->Release(); smaaSearchSRV = nullptr; }
    if (smaaVS) { smaaVS->Release(); smaaVS = nullptr; }
    if (smaaEdgePS) { smaaEdgePS->Release(); smaaEdgePS = nullptr; }
    if (smaaWeightPS) { smaaWeightPS->Release(); smaaWeightPS = nullptr; }
    if (smaaBlendPS) { smaaBlendPS->Release(); smaaBlendPS = nullptr; }
    if (smaaCB) { smaaCB->Release(); smaaCB = nullptr; }
    smaaEnabled = false;
}

BOOL MarniDX::Impl::ResizeSmaa()
{
    if (!smaaEnabled) return TRUE;
    if (smaaSceneSRV) { smaaSceneSRV->Release(); smaaSceneSRV = nullptr; }
    if (smaaSceneRTV) { smaaSceneRTV->Release(); smaaSceneRTV = nullptr; }
    if (smaaScene) { smaaScene->Release(); smaaScene = nullptr; }
    if (smaaEdgesSRV) { smaaEdgesSRV->Release(); smaaEdgesSRV = nullptr; }
    if (smaaEdgesRTV) { smaaEdgesRTV->Release(); smaaEdgesRTV = nullptr; }
    if (smaaEdges) { smaaEdges->Release(); smaaEdges = nullptr; }
    if (smaaWeightsSRV) { smaaWeightsSRV->Release(); smaaWeightsSRV = nullptr; }
    if (smaaWeightsRTV) { smaaWeightsRTV->Release(); smaaWeightsRTV = nullptr; }
    if (smaaWeights) { smaaWeights->Release(); smaaWeights = nullptr; }
    return MakeSmaaTarget(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
                          &smaaScene, &smaaSceneRTV, &smaaSceneSRV) &&
           MakeSmaaTarget(device, width, height, DXGI_FORMAT_R8G8_UNORM,
                          &smaaEdges, &smaaEdgesRTV, &smaaEdgesSRV) &&
           MakeSmaaTarget(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
                          &smaaWeights, &smaaWeightsRTV, &smaaWeightsSRV);
}

BOOL MarniDX::Impl::CreateSmaa()
{
    if (!g_bSmaa) return TRUE;
    char module[MAX_PATH + 32] = {};
    if (!GetModuleFileNameA(nullptr, module, MAX_PATH)) return FALSE;
    char* slash = strrchr(module, '\\');
    if (!slash) slash = strrchr(module, '/');
    if (!slash) return FALSE;
    slash[1] = '\0';
    strcat_s(module, sizeof(module), "smaa\\SMAA.hlsl");
    FILE* file = fopen(module, "rb");
    if (!file) {
        OutputDebugStringA("[SMAA] missing smaa/SMAA.hlsl; disabling SMAA\n");
        return FALSE;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    rewind(file);
    if (length <= 0) { fclose(file); return FALSE; }
    std::string source;
    source.resize((size_t)length);
    if (fread(&source[0], 1, source.size(), file) != source.size()) {
        fclose(file); return FALSE;
    }
    fclose(file);
    const char* sampleLevel = "SampleLevel(LinearSampler, coord, 0)";
    size_t sampleAt = source.find(sampleLevel);
    if (sampleAt != std::string::npos)
        source.replace(sampleAt, strlen(sampleLevel), "Sample(LinearSampler, coord)");
    sampleAt = source.find("SampleLevel(PointSampler, coord, 0)");
    if (sampleAt != std::string::npos)
        source.replace(sampleAt, strlen("SampleLevel(PointSampler, coord, 0)"), "Sample(PointSampler, coord)");
    sampleAt = source.find("SampleLevel(LinearSampler, coord, 0, offset)");
    if (sampleAt != std::string::npos)
        source.replace(sampleAt, strlen("SampleLevel(LinearSampler, coord, 0, offset)"), "Sample(LinearSampler, coord, offset)");
    // Wine's current D3DCompile implementation rejects HLSL flow-control
    // attributes even though they are optimization hints only.
    for (const char* attribute : { "[flatten]", "[branch]" }) {
        size_t at = 0;
        while ((at = source.find(attribute, at)) != std::string::npos)
            source.erase(at, strlen(attribute));
    }
    // The portable Wine compiler lacks discard/clip; edge pixels with zero
    // edges already produce the correct transparent intermediate value.
    size_t discardAt = 0;
    while ((discardAt = source.find("discard;", discardAt)) != std::string::npos) {
        source.replace(discardAt, 8, ";");
        discardAt += 1;
    }
    size_t loadBlock = source.find("#ifdef SMAALoad");
    if (loadBlock != std::string::npos)
        source.replace(loadBlock, strlen("#ifdef SMAALoad"), "#if 0");
    source.insert(0, "#define SMAA_HLSL_4\n#define SMAA_PRESET_HIGH\n"
        "#define mad(a,b,c) ((a)*(b)+(c))\n"
        "#define SMAA_RT_METRICS g_SmaaMetrics\n"
        "cbuffer SmaaCB : register(b0) { float4 g_SmaaMetrics; };\n");
    source += "\n#undef SMAALoad\n"; // this port uses 1x SMAA, never the optional MSAA pass
    source += g_SmaaWrappers;

    ID3DBlob *vs = nullptr, *edge = nullptr, *weight = nullptr, *blend = nullptr;
    if (!CompileSmaaShader(source, "SmaaVS", "vs_4_0", &vs) ||
        !CompileSmaaShader(source, "SmaaEdgePS", "ps_4_0", &edge) ||
        !CompileSmaaShader(source, "SmaaWeightPS", "ps_4_0", &weight) ||
        !CompileSmaaShader(source, "SmaaBlendPS", "ps_4_0", &blend)) {
        if (vs) vs->Release(); if (edge) edge->Release();
        if (weight) weight->Release(); if (blend) blend->Release();
        return FALSE;
    }
    HRESULT hr = device->CreateVertexShader(vs->GetBufferPointer(), vs->GetBufferSize(), nullptr, &smaaVS);
    if (SUCCEEDED(hr)) hr = device->CreatePixelShader(edge->GetBufferPointer(), edge->GetBufferSize(), nullptr, &smaaEdgePS);
    if (SUCCEEDED(hr)) hr = device->CreatePixelShader(weight->GetBufferPointer(), weight->GetBufferSize(), nullptr, &smaaWeightPS);
    if (SUCCEEDED(hr)) hr = device->CreatePixelShader(blend->GetBufferPointer(), blend->GetBufferSize(), nullptr, &smaaBlendPS);
    vs->Release(); edge->Release(); weight->Release(); blend->Release();
    if (FAILED(hr)) { ReleaseSmaa(); return FALSE; }

    D3D11_BUFFER_DESC cb = {};
    cb.ByteWidth = 16; cb.Usage = D3D11_USAGE_DYNAMIC;
    cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device->CreateBuffer(&cb, nullptr, &smaaCB))) { ReleaseSmaa(); return FALSE; }

    ID3D11Texture2D *area = nullptr, *search = nullptr;
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = AREATEX_WIDTH; td.Height = AREATEX_HEIGHT; td.MipLevels = 1;
    td.ArraySize = 1; td.Format = DXGI_FORMAT_R8G8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA init = {}; init.pSysMem = areaTexBytes; init.SysMemPitch = AREATEX_PITCH;
    hr = device->CreateTexture2D(&td, &init, &area);
    if (SUCCEEDED(hr)) hr = device->CreateShaderResourceView(area, nullptr, &smaaAreaSRV);
    if (area) area->Release();
    td.Width = SEARCHTEX_WIDTH; td.Height = SEARCHTEX_HEIGHT; td.Format = DXGI_FORMAT_R8_UNORM;
    init.pSysMem = searchTexBytes; init.SysMemPitch = SEARCHTEX_PITCH;
    if (SUCCEEDED(hr)) hr = device->CreateTexture2D(&td, &init, &search);
    if (SUCCEEDED(hr)) hr = device->CreateShaderResourceView(search, nullptr, &smaaSearchSRV);
    if (search) search->Release();
    if (FAILED(hr)) { ReleaseSmaa(); return FALSE; }
    smaaEnabled = true;
    if (!ResizeSmaa()) { ReleaseSmaa(); return FALSE; }
    OutputDebugStringA("[SMAA] SMAA 1x enabled\n");
    return TRUE;
}

void MarniDX::Impl::RenderSmaa()
{
    if (!smaaEnabled || !context || !rtv) return;
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (SUCCEEDED(context->Map(smaaCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        float* metrics = (float*)mapped.pData;
        metrics[0] = 1.0f / width; metrics[1] = 1.0f / height;
        metrics[2] = (float)width; metrics[3] = (float)height;
        context->Unmap(smaaCB, 0);
    }
    context->OMSetRenderTargets(0, nullptr, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width = (float)width; vp.Height = (float)height; vp.MaxDepth = 1.0f;
    context->RSSetViewports(1, &vp);
    D3D11_RECT rect = {0, 0, (LONG)width, (LONG)height};
    context->RSSetScissorRects(1, &rect);
    context->RSSetState(rasterScissor);
    context->OMSetBlendState(blendDisabled, nullptr, 0xffffffff);
    context->OMSetDepthStencilState(depthDisabled, 0);
    context->IASetInputLayout(nullptr);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(smaaVS, nullptr, 0);
    context->VSSetConstantBuffers(0, 1, &smaaCB);
    context->PSSetConstantBuffers(0, 1, &smaaCB);
    ID3D11ShaderResourceView* srvs[5] = {};

    context->OMSetRenderTargets(1, &smaaEdgesRTV, nullptr);
    srvs[0] = smaaSceneSRV; context->PSSetShaderResources(0, 1, srvs);
    context->PSSetShader(smaaEdgePS, nullptr, 0); context->Draw(3, 0);
    srvs[0] = nullptr; context->PSSetShaderResources(0, 1, srvs);

    context->OMSetRenderTargets(1, &smaaWeightsRTV, nullptr);
    srvs[1] = smaaEdgesSRV; srvs[2] = smaaAreaSRV; srvs[3] = smaaSearchSRV;
    context->PSSetShaderResources(1, 3, &srvs[1]);
    context->PSSetShader(smaaWeightPS, nullptr, 0); context->Draw(3, 0);
    srvs[1] = srvs[2] = srvs[3] = nullptr; context->PSSetShaderResources(1, 3, &srvs[1]);

    context->OMSetRenderTargets(1, &rtv, nullptr);
    srvs[0] = smaaSceneSRV; srvs[4] = smaaWeightsSRV;
    context->PSSetShaderResources(0, 1, srvs); context->PSSetShaderResources(4, 1, &srvs[4]);
    context->PSSetShader(smaaBlendPS, nullptr, 0); context->Draw(3, 0);
    srvs[0] = srvs[4] = nullptr; context->PSSetShaderResources(0, 1, srvs); context->PSSetShaderResources(4, 1, &srvs[4]);
    context->OMSetRenderTargets(1, &smaaSceneRTV, depthStencilView);
}

// Convert host bpp (4/8/16/24/32) to a contiguous RGBA8 (one DWORD/pixel)
// buffer suitable for DXGI_FORMAT_R8G8B8A8_UNORM upload.
// Returns a heap-allocated array (caller frees via free) or nullptr.
static DWORD* ConvertToRGBA8(int width, int height, int bpp,
                             const void* pixelData)
{
    if (!pixelData || width <= 0 || height <= 0) return nullptr;
    DWORD total = (DWORD)width * (DWORD)height;
    DWORD* rgba = (DWORD*)malloc(sizeof(DWORD) * total);
    if (!rgba) return nullptr;

    if (bpp == 32) {
        memcpy(rgba, pixelData, sizeof(DWORD) * total);
    } else if (bpp == 24) {
        BYTE* s = (BYTE*)pixelData;
        for (DWORD i = 0; i < total; i++) {
            rgba[i] = 0xFF000000u | (s[0]) | (s[1] << 8) | (s[2] << 16);
            s += 3;
        }
    } else if (bpp == 16) {
        WORD* s = (WORD*)pixelData;
        for (DWORD i = 0; i < total; i++) {
            WORD p = s[i];
            DWORD a = (p & 0x8000) ? 0xFF : 0x00;
            DWORD r = ((p >> 10) & 0x1F) * 255 / 31;
            DWORD g = ((p >> 5)  & 0x1F) * 255 / 31;
            DWORD b = ( p        & 0x1F) * 255 / 31;
            rgba[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    } else if (bpp == 8) {
        BYTE* s = (BYTE*)pixelData;
        for (DWORD i = 0; i < total; i++) {
            DWORD c = s[i];
            rgba[i] = 0xFF000000u | (c << 16) | (c << 8) | c;
        }
    } else if (bpp == 4) {
        WORD* s = (WORD*)pixelData;
        int words = total / 4;
        for (int i = 0; i < words; i++) {
            WORD w = s[i];
            DWORD p0 = ((w >> 0)  & 0xF) * 17;
            DWORD p1 = ((w >> 4)  & 0xF) * 17;
            DWORD p2 = ((w >> 8)  & 0xF) * 17;
            DWORD p3 = ((w >> 12) & 0xF) * 17;
            int b = i * 4;
            rgba[b]     = 0xFF000000u | (p0 << 16) | (p0 << 8) | p0;
            rgba[b + 1] = 0xFF000000u | (p1 << 16) | (p1 << 8) | p1;
            rgba[b + 2] = 0xFF000000u | (p2 << 16) | (p2 << 8) | p2;
            rgba[b + 3] = 0xFF000000u | (p3 << 16) | (p3 << 8) | p3;
        }
    } else {
        free(rgba);
        return nullptr;
    }
    return rgba;
}

// ----------------------------------------------------------------------------
// Impl helpers
// ----------------------------------------------------------------------------
void MarniDX::Impl::ReleaseSlot(int idx)
{
    if (idx <= 0 || idx >= MARNI_MAX_TEXTURES) return;
    TexSlot& t = slots[idx];
    if (t.srv) { t.srv->Release();   t.srv = nullptr; }
    if (t.tex) { t.tex->Release();   t.tex = nullptr; }
    t.width = 0; t.height = 0;
}

int MarniDX::Impl::AllocSlot(ID3D11Texture2D* t, ID3D11ShaderResourceView* s,
                             int w, int h)
{
    if (!t || !s) {
        if (t) t->Release();
        if (s) s->Release();
        return 0;
    }
    for (int i = 1; i < MARNI_MAX_TEXTURES; i++) {
        if (slots[i].tex == nullptr) {
            slots[i].tex   = t;
            slots[i].srv   = s;
            slots[i].width = w;
            slots[i].height = h;
            return i;
        }
    }
    // table full — reject.
    t->Release(); s->Release();
    return 0;
}

void MarniDX::Impl::ReleaseAllTextures()
{
    for (int i = 1; i < MARNI_MAX_TEXTURES; i++) ReleaseSlot(i);
    // white texture lives in slot? It's tracked by whiteTex/whiteSRV not slot.
    if (whiteSRV) { whiteSRV->Release(); whiteSRV = nullptr; }
    if (whiteTex) { whiteTex->Release(); whiteTex = nullptr; }
}

void MarniDX::Impl::ReleaseAllState()
{
    if (context) context->ClearState();
    ReleaseFrameInterpolation();
    ReleaseSmaa();
    if (quadLayout)    { quadLayout->Release();    quadLayout    = nullptr; }
    if (quadVS)        { quadVS->Release();        quadVS        = nullptr; }
    if (quadPS)        { quadPS->Release();        quadPS        = nullptr; }
    if (quadVB)        { quadVB->Release();        quadVB        = nullptr; }
    if (spriteCB)      { spriteCB->Release();      spriteCB      = nullptr; }
    if (rasterScissor) { rasterScissor->Release(); rasterScissor = nullptr; }
    if (blendAlpha)    { blendAlpha->Release();    blendAlpha    = nullptr; }
    if (blendAdd)      { blendAdd->Release();      blendAdd      = nullptr; }
    if (blendDisabled) { blendDisabled->Release(); blendDisabled = nullptr; }
    if (sampLinear)    { sampLinear->Release();    sampLinear    = nullptr; }
    if (sampPoint)     { sampPoint->Release();     sampPoint     = nullptr; }
    if (model3DLayout) { model3DLayout->Release(); model3DLayout = nullptr; }
    if (model3DVS)     { model3DVS->Release();     model3DVS     = nullptr; }
    if (model3DVB)     { model3DVB->Release();     model3DVB     = nullptr; }
    if (depthEnabled)  { depthEnabled->Release();  depthEnabled  = nullptr; }
    if (depthDisabled) { depthDisabled->Release(); depthDisabled = nullptr; }
    if (depthTestNoWrite) { depthTestNoWrite->Release(); depthTestNoWrite = nullptr; }
    if (depthStencilView){ depthStencilView->Release(); depthStencilView = nullptr; }
    if (depthStencil)    { depthStencil->Release();     depthStencil     = nullptr; }
    if (rtv)            { rtv->Release();              rtv              = nullptr; }
    if (context)        { context->Release();          context          = nullptr; }
    if (swapChain)      { swapChain->Release();        swapChain        = nullptr; }
    if (device)         { device->Release();           device           = nullptr; }
}

// ============================================================================
// Lifecycle
// ============================================================================
BOOL MarniDX::Create(HWND hWnd, int width, int height, BOOL fullScreen,
                     int* outWidth, int* outHeight)
{
    Impl* p = m_pImpl;
    if (!p) return FALSE;

    if (width  < 320) width  = 640;
    if (height < 240) height = 480;

    // Borderless-fullscreen presentation: NEVER create a DXGI exclusive
    // swap chain. Exclusive scanout bypasses DWM composition, which made
    // the GDI-drawn MCI FMV frames invisible (audio-only playback), its
    // asynchronous mode switches exposed the desktop behind resizing
    // windows after each FMV, and releasing an exclusive swap chain could
    // hang the process at exit leaving the panel black. A windowed
    // blt-model swap chain presented into a screen-sized WS_POPUP window
    // gives the same full-screen result without any of those problems;
    // DWM stretches the game-resolution back buffer over the monitor.
    if (fullScreen) {
        MONITORINFO mi = { sizeof(mi) };
        if (GetMonitorInfoA(MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST), &mi)) {
            // Done BEFORE the device exists so the resulting WM_SIZE lands
            // while ready==FALSE and cannot trigger a swap-chain resize.
            SetWindowPos(hWnd, HWND_TOPMOST,
                         mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top,
                         SWP_NOACTIVATE | SWP_FRAMECHANGED);
        }
    }

    DXGI_SWAP_CHAIN_DESC sc = {};
    sc.BufferCount        = 2;
    sc.BufferDesc.Width   = width;
    sc.BufferDesc.Height  = height;
    sc.BufferDesc.Format  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc.BufferDesc.RefreshRate.Numerator   = 60;
    sc.BufferDesc.RefreshRate.Denominator = 1;
    sc.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc.OutputWindow       = hWnd;
    sc.SampleDesc.Count   = 1;
    sc.Windowed           = TRUE;
    sc.Flags              = 0;

    D3D_FEATURE_LEVEL fls[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL got;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr,
        D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, fls, 3, D3D11_SDK_VERSION,
        &sc, &p->swapChain, &p->device, &got, &p->context);
    if (FAILED(hr)) {
        OutputDebugStringA("[MarniDX] Hardware D3D11 failed, trying WARP\n");
        // Drop any partially-created device before retrying.
        if (p->device)    { p->device->Release();    p->device    = nullptr; }
        if (p->context)   { p->context->Release();   p->context   = nullptr; }
        if (p->swapChain) { p->swapChain->Release(); p->swapChain = nullptr; }
        hr = D3D11CreateDeviceAndSwapChain(nullptr,
            D3D_DRIVER_TYPE_WARP, nullptr, 0, fls, 3, D3D11_SDK_VERSION,
            &sc, &p->swapChain, &p->device, &got, &p->context);
    }
    if (FAILED(hr)) {
        OutputDebugStringA("[MarniDX] D3D11 device creation FAILED\n");
        return FALSE;
    }
    OutputDebugStringA("[MarniDX] D3D11 device created OK\n");
    p->hWnd   = hWnd;
    p->width  = (DWORD)width;
    p->height = (DWORD)height;

    // back buffer -> RTV
    ID3D11Texture2D* bb = nullptr;
    hr = p->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb);
    if (SUCCEEDED(hr)) {
        hr = p->device->CreateRenderTargetView(bb, nullptr, &p->rtv);
        bb->Release();
    }
    if (FAILED(hr)) { p->ReleaseAllState(); return FALSE; }

    // depth/stencil
    D3D11_TEXTURE2D_DESC ds = {};
    ds.Width            = width;  ds.Height = height;
    ds.MipLevels        = 1;      ds.ArraySize = 1;
    ds.Format           = DXGI_FORMAT_D24_UNORM_S8_UINT;
    ds.SampleDesc.Count = 1;
    ds.Usage            = D3D11_USAGE_DEFAULT;
    ds.BindFlags        = D3D11_BIND_DEPTH_STENCIL;
    if (SUCCEEDED(p->device->CreateTexture2D(&ds, nullptr, &p->depthStencil)))
        p->device->CreateDepthStencilView(p->depthStencil, nullptr,
                                          &p->depthStencilView);

    // rasterizer (scissor on)
    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode         = D3D11_FILL_SOLID;
    rs.CullMode         = D3D11_CULL_NONE;
    rs.ScissorEnable    = TRUE;
    rs.DepthClipEnable  = TRUE;
    if (FAILED(p->device->CreateRasterizerState(&rs, &p->rasterScissor))) {
        p->ReleaseAllState(); return FALSE;
    }

    // alpha blend
    D3D11_BLEND_DESC bd = {};
    bd.RenderTarget[0].BlendEnable        = TRUE;
    bd.RenderTarget[0].SrcBlend           = D3D11_BLEND_SRC_ALPHA;
    bd.RenderTarget[0].DestBlend          = D3D11_BLEND_INV_SRC_ALPHA;
    bd.RenderTarget[0].BlendOp            = D3D11_BLEND_OP_ADD;
    bd.RenderTarget[0].SrcBlendAlpha      = D3D11_BLEND_ONE;
    bd.RenderTarget[0].DestBlendAlpha     = D3D11_BLEND_INV_SRC_ALPHA;
    bd.RenderTarget[0].BlendOpAlpha       = D3D11_BLEND_OP_ADD;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(p->device->CreateBlendState(&bd, &p->blendAlpha))) {
        p->ReleaseAllState(); return FALSE;
    }

    // additive blend (used by MARNI_BLEND_ADD)
    bd.RenderTarget[0].SrcBlend  = D3D11_BLEND_ONE;
    bd.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
    p->device->CreateBlendState(&bd, &p->blendAdd);

    // disabled blend (straight overwrite)
    D3D11_BLEND_DESC bdOff = {};
    bdOff.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    p->device->CreateBlendState(&bdOff, &p->blendDisabled);

    // samplers
    D3D11_SAMPLER_DESC ss = {};
    ss.Filter       = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    ss.AddressU = ss.AddressV = ss.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    ss.ComparisonFunc = D3D11_COMPARISON_NEVER;
    ss.MaxLOD        = D3D11_FLOAT32_MAX;
    if (FAILED(p->device->CreateSamplerState(&ss, &p->sampLinear))) {
        p->ReleaseAllState(); return FALSE;
    }
    ss.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    if (FAILED(p->device->CreateSamplerState(&ss, &p->sampPoint))) {
        p->ReleaseAllState(); return FALSE;
    }

    // depth disabled (2D)
    D3D11_DEPTH_STENCIL_DESC dd = {};
    dd.DepthEnable    = FALSE;
    dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    dd.DepthFunc      = D3D11_COMPARISON_ALWAYS;
    if (FAILED(p->device->CreateDepthStencilState(&dd, &p->depthDisabled))) {
        p->ReleaseAllState(); return FALSE;
    }

    // depth enabled (3D models) - the 2D layers neither test nor write, so
    // enabling it here only affects the TMD path.
    D3D11_DEPTH_STENCIL_DESC de = {};
    de.DepthEnable    = TRUE;
    de.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    de.DepthFunc      = D3D11_COMPARISON_LESS_EQUAL;
    if (FAILED(p->device->CreateDepthStencilState(&de, &p->depthEnabled))) {
        p->ReleaseAllState(); return FALSE;
    }

    // depth test WITHOUT write - translucent world quads (the ground shadow /
    // blood pool fade polys). They must clip against the model geometry the
    // way the original's Z-buffered viewport quads did, but as alpha-blended
    // primitives they must not poison the buffer for the painter-ordered
    // draws around them.
    D3D11_DEPTH_STENCIL_DESC dt = {};
    dt.DepthEnable    = TRUE;
    dt.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    dt.DepthFunc      = D3D11_COMPARISON_LESS_EQUAL;
    if (FAILED(p->device->CreateDepthStencilState(&dt, &p->depthTestNoWrite))) {
        p->ReleaseAllState(); return FALSE;
    }

    // shaders
    if (!CompileShaders(p->device, &p->quadVS, &p->quadPS, &p->quadLayout)) {
        OutputDebugStringA("[MarniDX] shader compile failed\n");
        p->ReleaseAllState(); return FALSE;
    }
    if (!CompileModel3DShader(p->device, &p->model3DVS, &p->model3DLayout)) {
        OutputDebugStringA("[MarniDX] model shader compile failed\n");
        p->ReleaseAllState(); return FALSE;
    }

    // vertex buffer (6 verts/sprite * 512 sprites)
    D3D11_BUFFER_DESC vb = {};
    vb.Usage          = D3D11_USAGE_DYNAMIC;
    vb.ByteWidth      = sizeof(QuadVertex) * 6 * 512;
    vb.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    vb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(p->device->CreateBuffer(&vb, nullptr, &p->quadVB))) {
        p->ReleaseAllState(); return FALSE;
    }

    // 3D model vertex buffer (3 verts/tri * 1024 triangles per call)
    D3D11_BUFFER_DESC mvb = {};
    mvb.Usage          = D3D11_USAGE_DYNAMIC;
    mvb.ByteWidth      = sizeof(Model3DVertex) * 3 * 1024;
    mvb.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    mvb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(p->device->CreateBuffer(&mvb, nullptr, &p->model3DVB))) {
        p->ReleaseAllState(); return FALSE;
    }

    // sprite constant buffer
    D3D11_BUFFER_DESC cb = {};
    cb.Usage          = D3D11_USAGE_DYNAMIC;
    cb.ByteWidth      = sizeof(SpriteConstantBuffer);
    cb.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(p->device->CreateBuffer(&cb, nullptr, &p->spriteCB))) {
        p->ReleaseAllState(); return FALSE;
    }

    // white 1x1 texture
    D3D11_TEXTURE2D_DESC wt = {};
    wt.Width    = 1; wt.Height = 1; wt.MipLevels = 1; wt.ArraySize = 1;
    wt.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    wt.SampleDesc.Count = 1;
    wt.Usage    = D3D11_USAGE_IMMUTABLE;
    wt.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    DWORD whitePixel = 0xFFFFFFFFu;
    D3D11_SUBRESOURCE_DATA wi = {};
    wi.pSysMem     = &whitePixel;
    wi.SysMemPitch = 4;
    if (SUCCEEDED(p->device->CreateTexture2D(&wt, &wi, &p->whiteTex))) {
        D3D11_SHADER_RESOURCE_VIEW_DESC wsd = {};
        wsd.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
        wsd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        wsd.Texture2D.MipLevels = 1;
        p->device->CreateShaderResourceView(p->whiteTex, &wsd, &p->whiteSRV);
    }

    // SMAA is an optional final pass; a missing shader asset falls back to the
    // original direct-to-backbuffer path instead of preventing the game boot.
    if (!p->CreateSmaa())
        OutputDebugStringA("[SMAA] initialization failed; using normal D3D11 output\n");
    if (!p->CreateFrameInterpolation())
        OutputDebugStringA("[FrameInterpolation] initialization failed; keeping 30 Hz presentation\n");

    // bind game target + set viewport/scissor
    ID3D11RenderTargetView* gameTarget = p->smaaEnabled ? p->smaaSceneRTV : p->rtv;
    p->context->OMSetRenderTargets(1, &gameTarget, p->depthStencilView);
    D3D11_VIEWPORT vp = {};
    vp.Width = (float)p->width; vp.Height = (float)p->height; vp.MaxDepth = 1.0f;
    p->context->RSSetViewports(1, &vp);
    D3D11_RECT sr = { 0, 0, (LONG)p->width, (LONG)p->height };
    p->context->RSSetScissorRects(1, &sr);
    p->context->RSSetState(p->rasterScissor);

    p->ready = TRUE;

    // In borderless fullscreen, render at the monitor's native resolution.
    // The game-space -> backbuffer scale (MarniGetRenderScale = physical/
    // logical) is applied at draw time with POINT sampling, so a native-
    // resolution backbuffer keeps the 2D art crisp. Leaving the backbuffer
    // at the game resolution would make DWM stretch it over the screen-
    // sized window with bilinear filtering, blurring every glyph.
    if (fullScreen) {
        MONITORINFO mi = { sizeof(mi) };
        if (GetMonitorInfoA(MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST), &mi)) {
            int monW = mi.rcMonitor.right - mi.rcMonitor.left;
            int monH = mi.rcMonitor.bottom - mi.rcMonitor.top;
            if (monW >= 320 && monH >= 240 &&
                (monW != (int)p->width || monH != (int)p->height)) {
                ChangeDisplayMode((DWORD)monW, (DWORD)monH, FALSE);
            }
        }
    }

    if (outWidth)  *outWidth  = (int)p->width;
    if (outHeight) *outHeight = (int)p->height;
    return TRUE;
}

// ----------------------------------------------------------------------------
// Local helpers (file-local, all D3D11 stays in this TU)
// =============================================================================

void MarniDX::Destroy()
{
    if (!m_pImpl) return;
    m_pImpl->ReleaseAllTextures();
    m_pImpl->ReleaseAllState();
    m_pImpl->ready = FALSE;
    m_pImpl->fontHandle = MARNI_NULL_HANDLE;
    m_pImpl->fontW = 0;
    m_pImpl->fontH = 0;
}


BOOL MarniDX::IsReady() const { return m_pImpl && m_pImpl->ready; }

void MarniDX::GetBackBufferSize(DWORD* w, DWORD* h) const
{
    if (w) *w = m_pImpl ? m_pImpl->width  : 0;
    if (h) *h = m_pImpl ? m_pImpl->height : 0;
}

int MarniDX::ChangeDisplayMode(DWORD newW, DWORD newH, BOOL fullScreen)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready || !p->swapChain || !p->context) return 0;
    if (newW == 0 || newH == 0) return 0;
    if (newW < 320) newW = 320;
    if (newH < 240) newH = 240;

    p->context->OMSetRenderTargets(0, nullptr, nullptr);
    ID3D11ShaderResourceView* nullSrvs[5] = {};
    p->context->PSSetShaderResources(0, 5, nullSrvs);
    if (p->rtv) { p->rtv->Release(); p->rtv = nullptr; }
    // The depth surface has to be resized with the render target: D3D11 rejects
    // a depth view whose dimensions differ from the colour view, which would
    // silently leave the 3D models with no depth buffer after a resize.
    if (p->depthStencilView) { p->depthStencilView->Release(); p->depthStencilView = nullptr; }
    if (p->depthStencil)     { p->depthStencil->Release();     p->depthStencil     = nullptr; }

    HRESULT hr = p->swapChain->ResizeBuffers(2, newW, newH,
        DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    if (FAILED(hr)) return 0;

    p->width = newW; p->height = newH;
    if (p->frameInterpolation && !p->ResizeFrameInterpolation())
        p->ReleaseFrameInterpolation();
    if (p->smaaEnabled && !p->ResizeSmaa()) p->ReleaseSmaa();
    ID3D11Texture2D* bb = nullptr;
    hr = p->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb);
    if (SUCCEEDED(hr)) {
        p->device->CreateRenderTargetView(bb, nullptr, &p->rtv);
        bb->Release();
    }

    D3D11_TEXTURE2D_DESC ds = {};
    ds.Width            = newW;  ds.Height = newH;
    ds.MipLevels        = 1;     ds.ArraySize = 1;
    ds.Format           = DXGI_FORMAT_D24_UNORM_S8_UINT;
    ds.SampleDesc.Count = 1;
    ds.Usage            = D3D11_USAGE_DEFAULT;
    ds.BindFlags        = D3D11_BIND_DEPTH_STENCIL;
    if (SUCCEEDED(p->device->CreateTexture2D(&ds, nullptr, &p->depthStencil)))
        p->device->CreateDepthStencilView(p->depthStencil, nullptr,
                                          &p->depthStencilView);

    D3D11_VIEWPORT vp = {};
    vp.Width = (float)newW; vp.Height = (float)newH; vp.MaxDepth = 1.0f;
    p->context->RSSetViewports(1, &vp);
    D3D11_RECT sr = { 0, 0, (LONG)newW, (LONG)newH };
    p->context->RSSetScissorRects(1, &sr);
    ID3D11RenderTargetView* gameTarget = p->smaaEnabled ? p->smaaSceneRTV : p->rtv;
    p->context->OMSetRenderTargets(1, &gameTarget, p->depthStencilView);
    return 1;
}

int MarniDX::HandleWindowMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready) return 1;
    switch (msg) {
    case WM_SIZE:
        if (!p->swapChain || LOWORD(lParam) == 0) return 1;
        ChangeDisplayMode(LOWORD(lParam), HIWORD(lParam), FALSE);
        return 1;
    default:
        return 1;
    }
    (void)hwnd; (void)wParam;
}

unsigned int MarniDX::QueryVideoMemory(BOOL softwareRenderer) const
{
    if (softwareRenderer) return 0x75bcd15u;
    Impl* p = m_pImpl;
    if (!p || !p->device) return 128u * 1024u * 1024u;

    IDXGIDevice* dxDev = nullptr;
    if (SUCCEEDED(p->device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxDev))) {
        IDXGIAdapter* ad = nullptr;
        if (SUCCEEDED(dxDev->GetAdapter(&ad))) {
            DXGI_ADAPTER_DESC d;
            HRESULT hr = ad->GetDesc(&d);
            ad->Release();
            dxDev->Release();
            if (SUCCEEDED(hr)) return (unsigned int)(d.DedicatedVideoMemory & 0x7FFFFFFFu);
        } else { dxDev->Release(); }
    }
    return 128u * 1024u * 1024u;
}

// ============================================================================
// Display mode / adapter enumeration (DXGI remains inside this TU)
// ============================================================================
void MarniDX::EnumerateDisplayModes(DisplayModeInfo* outModes, int maxModes,
                                    int* outCount)
{
    Impl* p = m_pImpl;
    int count = 0;
    if (p && p->device && outModes && maxModes > 0) {
        IDXGIDevice* dxDev = nullptr;
        if (SUCCEEDED(p->device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxDev))) {
            IDXGIAdapter* ad = nullptr;
            if (SUCCEEDED(dxDev->GetAdapter(&ad))) {
                IDXGIOutput* out = nullptr;
                if (SUCCEEDED(ad->EnumOutputs(0, &out))) {
                    UINT num = 0;
                    out->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &num, nullptr);
                    if (num > 0) {
                        DXGI_MODE_DESC* modes = new DXGI_MODE_DESC[num];
                        out->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &num, modes);
                        for (UINT i = 0; i < num && count < maxModes; i++) {
                            bool dup = false;
                            for (int j = 0; j < count; j++)
                                if (outModes[j].dwWidth  == modes[i].Width &&
                                    outModes[j].dwHeight == modes[i].Height)
                                { dup = true; break; }
                            if (dup) continue;
                            outModes[count].dwWidth  = modes[i].Width;
                            outModes[count].dwHeight = modes[i].Height;
                            outModes[count].dwBPP    = 32;
                            outModes[count].dwRefreshRate =
                                modes[i].RefreshRate.Numerator /
                                (modes[i].RefreshRate.Denominator ? modes[i].RefreshRate.Denominator : 1);
                            outModes[count].dwFlags = 1; // fullscreen
                            count++;
                        }
                        delete[] modes;
                    }
                    out->Release();
                }
                ad->Release();
            }
            dxDev->Release();
        }
    }
    if (count == 0) {
        // fallback defaults
        const DWORD defs[5][3] = {
            { 640, 480, 0 },  { 800, 600, 1 },  { 1024, 768, 1 },
            { 1280, 720, 1 }, { 1920, 1080, 1 },
        };
        for (int i = 0; i < 5 && i < maxModes; i++) {
            outModes[i].dwWidth = defs[i][0];
            outModes[i].dwHeight = defs[i][1];
            outModes[i].dwBPP    = 32;
            outModes[i].dwRefreshRate = 60;
            outModes[i].dwFlags  = defs[i][2];
            count++;
        }
    }
    if (outCount) *outCount = count;
}

void MarniDX::EnumerateAdapters(D3DRendererInfo* out, int max, int* outCount)
{
    int count = 0;
    IDXGIFactory* fac = nullptr;
    if (SUCCEEDED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&fac))) {
        IDXGIAdapter* ad = nullptr;
        for (UINT i = 0; fac->EnumAdapters(i, &ad) != DXGI_ERROR_NOT_FOUND
                          && count < max; i++) {
            DXGI_ADAPTER_DESC d;
            if (SUCCEEDED(ad->GetDesc(&d)) && out) {
                WideCharToMultiByte(CP_ACP, 0, d.Description, -1,
                    out[count].name, 256, nullptr, nullptr);
                out[count].flags = 1;
                count++;
            }
            ad->Release();
        }
        fac->Release();
    }
    if (count == 0 && out && max > 0) {
        strcpy_s(out[0].name, 256, "Primary Display Adapter");
        out[0].flags = 1;
        count = 1;
    }
    if (outCount) *outCount = count;
}

// ============================================================================
// Per-frame
// ============================================================================
void MarniDX::Clear(float r, float g, float b, float a)
{
    Impl* p = m_pImpl;
    if (!p || !p->context || !p->rtv) return;
    // Presentation modes can leave bars outside the configured viewport. Keep
    // those deterministic during fades: the authored frame paints its own
    // opaque background, while a coloured ClearRenderTargetView would make the
    // side bars flash red/white with title and transition clear colours.
    if (p->presentation != MARNI_PRESENT_STRETCH) {
        r = g = b = 0.0f;
        a = 1.0f;
    }
    float c[4] = { r, g, b, a };
    p->context->ClearRenderTargetView(p->smaaEnabled ? p->smaaSceneRTV : p->rtv, c);
    if (p->depthStencilView)
        p->context->ClearDepthStencilView(p->depthStencilView,
            D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
}

void MarniDX::SetPresentation(MarniPresentationMode mode, float verticalPan)
{
    if (!m_pImpl) return;
    m_pImpl->presentation = mode;
    m_pImpl->verticalPan = verticalPan;
    if (m_pImpl->context) {
        const D3D11_RECT rect = BuildPresentationScissor(m_pImpl);
        m_pImpl->context->RSSetScissorRects(1, &rect);
    }
}

// Opt-in scripted-test hook (not in the original): RE1_TEST_SHOT_EVERY=N saves
// every Nth presented frame as shotNNNN.ppm in the working directory, up to
// RE1_TEST_SHOT_MAX (default 40). Scripted runs cannot press PrintScreen
// because test keys bypass the window's message loop.
static void SaveTestShot(MarniDX* dx)
{
    static const int every = std::getenv("RE1_TEST_SHOT_EVERY")
        ? std::atoi(std::getenv("RE1_TEST_SHOT_EVERY")) : 0;
    static const int maxShots = std::getenv("RE1_TEST_SHOT_MAX")
        ? std::atoi(std::getenv("RE1_TEST_SHOT_MAX")) : 40;
    static int frame = 0, saved = 0;
    if (every <= 0 || saved >= maxShots || (++frame % every) != 0) return;
    void* pixels = nullptr;
    DWORD w = 0, h = 0;
    if (!dx->CaptureBackbufferToRGBA(&pixels, &w, &h) || !pixels) return;
    char name[32];
    snprintf(name, sizeof(name), "shot%04d.ppm", saved++);
    if (FILE* f = fopen(name, "wb")) {
        fprintf(f, "P6\n%lu %lu\n255\n", (unsigned long)w, (unsigned long)h);
        const BYTE* px = (const BYTE*)pixels;
        for (DWORD i = 0; i < w * h; ++i) fwrite(px + i * 4, 1, 3, f);
        fclose(f);
    }
    free(pixels);
}

void MarniDX::Present()
{
    Impl* p = m_pImpl;
    if (!p || !p->swapChain) return;

    if (p->smaaEnabled) p->RenderSmaa();
    SaveTestShot(this);

    if (p->frameInterpolation) {
        // Pose interpolation renders two complete scene passes per 30 Hz
        // simulation tick. Each one consumes one refresh; do not blend the
        // completed images here, because that ghosts moving actors and gives
        // the same pixels to both presentation slots.
        p->swapChain->Present(1, 0);
        ID3D11RenderTargetView* gameTarget = p->smaaEnabled ? p->smaaSceneRTV : p->rtv;
        p->context->OMSetRenderTargets(1, &gameTarget, p->depthStencilView);
        return;
    }

    // Do NOT wait on a vblank by default. This engine paces itself in software:
    // the pump limits main_loop to one call per 33 ms (g_dwFrameIntervalMs) and
    // FrameRateGovernor decides which of those frames get presented. A blocking
    // Present puts the display in charge of both instead, and the two fight:
    // with SyncInterval 1 on a 60 Hz panel the real frame period becomes 33.33
    // ms, which the governor measures as 34 often enough to push its target to
    // 101 - inside the 101..103 notch the 104/132 scaling does not cover - which
    // sets g_bFrameSkipDetected, which DISABLES the 33 ms limiter, which lets a
    // burst of frames run at the full 16.7 ms refresh before the average settles
    // and pacing resumes. Measured: 31.6 ms per tick instead of 33, i.e. ~4%
    // fast, oscillating. The original blitted to the window without waiting for
    // a vblank, so its limiter always won.
    p->swapChain->Present(g_bVSync ? 1 : 0, 0);
    ID3D11RenderTargetView* gameTarget = p->smaaEnabled ? p->smaaSceneRTV : p->rtv;
    if (gameTarget) p->context->OMSetRenderTargets(1, &gameTarget, p->depthStencilView);
}

BOOL MarniDX::CapturePoseInterpolationBase()
{
    Impl* p = m_pImpl;
    if (!p || !p->frameInterpolation || !p->frameCurrent || !p->context)
        return FALSE;

    ID3D11Texture2D* source = p->smaaEnabled ? p->smaaScene : nullptr;
    ID3D11Texture2D* backBuffer = nullptr;
    if (source == nullptr) {
        if (FAILED(p->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                           (void**)&backBuffer))) return FALSE;
        source = backBuffer;
    }
    p->context->OMSetRenderTargets(0, nullptr, nullptr);
    p->context->CopyResource(p->frameCurrent, source);
    if (backBuffer) backBuffer->Release();
    ID3D11RenderTargetView* gameTarget = p->smaaEnabled ? p->smaaSceneRTV : p->rtv;
    p->context->OMSetRenderTargets(1, &gameTarget, p->depthStencilView);
    return TRUE;
}

BOOL MarniDX::RestorePoseInterpolationBase()
{
    Impl* p = m_pImpl;
    if (!p || !p->frameInterpolation || !p->frameCurrent || !p->context)
        return FALSE;

    ID3D11Texture2D* destination = p->smaaEnabled ? p->smaaScene : nullptr;
    ID3D11Texture2D* backBuffer = nullptr;
    if (destination == nullptr) {
        if (FAILED(p->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D),
                                           (void**)&backBuffer))) return FALSE;
        destination = backBuffer;
    }
    p->context->OMSetRenderTargets(0, nullptr, nullptr);
    p->context->CopyResource(destination, p->frameCurrent);
    if (backBuffer) backBuffer->Release();
    if (p->depthStencilView)
        p->context->ClearDepthStencilView(p->depthStencilView,
            D3D11_CLEAR_DEPTH, 1.0f, 0);
    ID3D11RenderTargetView* gameTarget = p->smaaEnabled ? p->smaaSceneRTV : p->rtv;
    p->context->OMSetRenderTargets(1, &gameTarget, p->depthStencilView);
    return TRUE;
}

// ============================================================================
// Texture handles
// ============================================================================
MarniHandle MarniDX::CreateTexture(int width, int height, int bpp,
                                   const void* pixelData,
                                   int* outWidth, int* outHeight)
{
    Impl* p = m_pImpl;
    if (!p || !p->device || width <= 0 || height <= 0 || !pixelData)
        return MARNI_NULL_HANDLE;

    DWORD* rgba = ConvertToRGBA8(width, height, bpp, pixelData);
    if (!rgba) return MARNI_NULL_HANDLE;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width           = (UINT)width;
    td.Height          = (UINT)height;
    td.MipLevels       = 1;
    td.ArraySize       = 1;
    td.Format          = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage           = D3D11_USAGE_DEFAULT;
    td.BindFlags       = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd = {};
    sd.pSysMem     = rgba;
    sd.SysMemPitch = (UINT)width * 4;

    ID3D11Texture2D* tex = nullptr;
    HRESULT hr = p->device->CreateTexture2D(&td, &sd, &tex);
    free(rgba);
    if (FAILED(hr) || !tex) return MARNI_NULL_HANDLE;

    D3D11_SHADER_RESOURCE_VIEW_DESC sdv = {};
    sdv.Format        = DXGI_FORMAT_R8G8B8A8_UNORM;
    sdv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sdv.Texture2D.MipLevels = 1;
    ID3D11ShaderResourceView* srv = nullptr;
    hr = p->device->CreateShaderResourceView(tex, &sdv, &srv);
    if (FAILED(hr) || !srv) { tex->Release(); return MARNI_NULL_HANDLE; }

    int idx = p->AllocSlot(tex, srv, width, height);
    if (idx == 0) return MARNI_NULL_HANDLE;
    if (outWidth)  *outWidth  = width;
    if (outHeight) *outHeight = height;
    return (MarniHandle)idx;
}

MarniHandle MarniDX::CreateTextureFromBits(CMarniBits* bits)
{
    if (!bits) return MARNI_NULL_HANDLE;
    // The bits surface is RGBA host memory laid out in m_pitch rows. We
    // re-interleave via the existing ConvertToRGBA8 path by treating the
    // surface as 32-bit RGBA.
    int w = (int)bits->m_width;
    int h = (int)bits->m_height;
    if (w <= 0 || h <= 0 || bits->m_bitDepth != 32) return MARNI_NULL_HANDLE;
    return CreateTexture(w, h, 32, bits->m_pPixelData, nullptr, nullptr);
}

BOOL MarniDX::UpdateTexturePixels(MarniHandle tex, const void* pixelData,
                                  int width, int height, int bpp)
{
    Impl* p = m_pImpl;
    if (!p || !p->context || !pixelData) return FALSE;
    if (tex <= 0 || tex >= MARNI_MAX_TEXTURES) return FALSE;
    TexSlot& t = p->slots[tex];
    if (!t.tex) return FALSE;
    if (width != t.width || height != t.height) return FALSE;

    // Convert the source to RGBA8 and upload via an staging update subrect.
    DWORD* rgba = ConvertToRGBA8(width, height, bpp, pixelData);
    if (!rgba) return FALSE;

    // UpdateSubresource expects contiguous rows of the source format. Since
    // our texture is already R8G8B8A8, write the converted buffer directly.
    D3D11_BOX dstBox = { 0, 0, 0, (UINT)width, (UINT)height, 1 };
    p->context->UpdateSubresource(t.tex, 0, &dstBox, rgba, (UINT)width * 4, 0);
    free(rgba);
    return TRUE;
}

void MarniDX::GetTextureSize(MarniHandle tex, int* w, int* h) const
{
    Impl* p = m_pImpl;
    if (!p || tex <= 0 || tex >= MARNI_MAX_TEXTURES) { if (w)*w=0; if(h)*h=0; return; }
    const TexSlot& t = p->slots[tex];
    if (w) *w = t.width;
    if (h) *h = t.height;
}

void MarniDX::DestroyTexture(MarniHandle tex)
{
    Impl* p = m_pImpl;
    if (!p) return;
    if ((int)tex <= 0 || (int)tex >= MARNI_MAX_TEXTURES) return;
    p->ReleaseSlot((int)tex);
    // Clear font cache if it pointed here.
    if (p->fontHandle == tex) {
        p->fontHandle = MARNI_NULL_HANDLE; p->fontW = 0; p->fontH = 0;
    }
}

MarniHandle MarniDX::WhiteTexture() const
{
    // DrawQuadInternal already maps MARNI_NULL_HANDLE to the internal white
    // SRV, so callers that need an explicit "white" handle can pass zero and
    // get the same result. This accessor returns the sentinel directly.
    return MARNI_NULL_HANDLE;
}

// ============================================================================
// Drawing
// ============================================================================
static void DrawQuadInternal(MarniDX::Impl* p, const QuadVertex verts[6],
                             MarniHandle tex, MarniSampler sampler, MarniBlend blend)
{
    if (!p || !p->context || !p->quadVB || !p->spriteCB || !p->quadVS
        || !p->quadPS || !p->quadLayout) return;

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(p->context->Map(p->quadVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return;
    memcpy(m.pData, verts, sizeof(QuadVertex) * 6);
    p->context->Unmap(p->quadVB, 0);

    SpriteConstantBuffer cb;
    // Y-down ortho (screen-space): origin top-left, +Y goes down.
    // bottom=height, top=0 so that pos.y=0 -> NDC +1 (top) and
    // pos.y=height -> NDC -1 (bottom). Without this swap the image
    // is rendered vertically flipped.
    BuildPresentationMatrix(&cb.mvp[0][0], p);
    D3D11_MAPPED_SUBRESOURCE cm = {};
    if (SUCCEEDED(p->context->Map(p->spriteCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &cm))) {
        memcpy(cm.pData, &cb, sizeof(cb));
        p->context->Unmap(p->spriteCB, 0);
    }

    UINT stride = sizeof(QuadVertex), offset = 0;
    p->context->IASetVertexBuffers(0, 1, &p->quadVB, &stride, &offset);
    p->context->IASetInputLayout(p->quadLayout);
    p->context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    p->context->VSSetShader(p->quadVS, nullptr, 0);
    p->context->VSSetConstantBuffers(0, 1, &p->spriteCB);
    p->context->PSSetShader(p->quadPS, nullptr, 0);

    ID3D11ShaderResourceView* srv = nullptr;
    if ((int)tex > 0 && (int)tex < MARNI_MAX_TEXTURES)
        srv = p->slots[tex].srv;
    if (!srv) srv = p->whiteSRV;
    if (!srv) return; // no fallback at all -- skip
    p->context->PSSetShaderResources(0, 1, &srv);

    ID3D11SamplerState* s = (sampler == MARNI_SAMPLER_POINT) ? p->sampPoint : p->sampLinear;
    if (!s) s = p->sampLinear;
    p->context->PSSetSamplers(0, 1, &s);

    ID3D11BlendState* bs = p->blendAlpha;
    if (blend == MARNI_BLEND_ADD)       bs = p->blendAdd;
    else if (blend == MARNI_BLEND_DISABLE) bs = p->blendDisabled;
    if (!bs) bs = p->blendAlpha;
    float bf[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    p->context->OMSetBlendState(bs, bf, 0xFFFFFFFFu);

    if (p->depthDisabled)
        p->context->OMSetDepthStencilState(p->depthDisabled, 0);

    p->context->Draw(6, 0);
}

void MarniDX::DrawSprite(float x, float y, float w, float h,
                         float u0, float v0, float u1, float v1,
                         DWORD color, MarniHandle tex,
                         MarniSampler sampler, MarniBlend blend)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready) return;
    float r = ((color >> 16) & 0xFF) / 255.0f;
    float g = ((color >> 8)  & 0xFF) / 255.0f;
    float b = ( color        & 0xFF) / 255.0f;
    float a = ((color >> 24) & 0xFF) / 255.0f;

    float x1 = x + w, y1 = y + h;
    QuadVertex verts[6] = {
        { x,  y,  u0, v0, r, g, b, a },
        { x1, y,  u1, v0, r, g, b, a },
        { x,  y1, u0, v1, r, g, b, a },
        { x,  y1, u0, v1, r, g, b, a },
        { x1, y,  u1, v0, r, g, b, a },
        { x1, y1, u1, v1, r, g, b, a },
    };
    DrawQuadInternal(p, verts, tex, sampler, blend);
}

void MarniDX::DrawTriangles(const float* verts, int triCount, MarniHandle tex,
                            MarniSampler sampler, MarniBlend blend)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready || !verts || triCount <= 0) return;
    // quadVB capacity: 6 verts * 512 sprites = 3072 verts = 1024 triangles
    if (triCount > 1024) triCount = 1024;
    // DrawTriangles shares the quad pipeline: same vertex layout, same ortho
    // MVP, but an arbitrary triangle count instead of the fixed 2-tri quad.
    if (!p->context || !p->quadVB || !p->spriteCB || !p->quadVS
        || !p->quadPS || !p->quadLayout) return;

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(p->context->Map(p->quadVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return;
    memcpy(m.pData, verts, sizeof(QuadVertex) * 3 * (size_t)triCount);
    p->context->Unmap(p->quadVB, 0);

    SpriteConstantBuffer cb;
    BuildPresentationMatrix(&cb.mvp[0][0], p);
    D3D11_MAPPED_SUBRESOURCE cm = {};
    if (SUCCEEDED(p->context->Map(p->spriteCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &cm))) {
        memcpy(cm.pData, &cb, sizeof(cb));
        p->context->Unmap(p->spriteCB, 0);
    }

    UINT stride = sizeof(QuadVertex), offset = 0;
    p->context->IASetVertexBuffers(0, 1, &p->quadVB, &stride, &offset);
    p->context->IASetInputLayout(p->quadLayout);
    p->context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    p->context->VSSetShader(p->quadVS, nullptr, 0);
    p->context->VSSetConstantBuffers(0, 1, &p->spriteCB);
    p->context->PSSetShader(p->quadPS, nullptr, 0);

    ID3D11ShaderResourceView* srv = nullptr;
    if ((int)tex > 0 && (int)tex < MARNI_MAX_TEXTURES)
        srv = p->slots[tex].srv;
    if (!srv) srv = p->whiteSRV;
    if (!srv) return;
    p->context->PSSetShaderResources(0, 1, &srv);

    ID3D11SamplerState* s = (sampler == MARNI_SAMPLER_POINT) ? p->sampPoint : p->sampLinear;
    if (!s) s = p->sampLinear;
    p->context->PSSetSamplers(0, 1, &s);

    ID3D11BlendState* bs = p->blendAlpha;
    if (blend == MARNI_BLEND_ADD)       bs = p->blendAdd;
    else if (blend == MARNI_BLEND_DISABLE) bs = p->blendDisabled;
    if (!bs) bs = p->blendAlpha;
    float bf[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    p->context->OMSetBlendState(bs, bf, 0xFFFFFFFFu);

    if (p->depthDisabled)
        p->context->OMSetDepthStencilState(p->depthDisabled, 0);

    p->context->Draw((UINT)(triCount * 3), 0);
}

void MarniDX::DrawTriangles3D(const float* verts, int triCount, MarniHandle tex,
                              MarniSampler sampler, MarniBlend blend,
                              bool depthWrite)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready || !verts || triCount <= 0) return;
    if (triCount > 1024) triCount = 1024;
    if (!p->context || !p->model3DVB || !p->spriteCB || !p->model3DVS
        || !p->quadPS || !p->model3DLayout) return;

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(p->context->Map(p->model3DVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return;
    memcpy(m.pData, verts, sizeof(Model3DVertex) * 3 * (size_t)triCount);
    p->context->Unmap(p->model3DVB, 0);

    SpriteConstantBuffer cb;
    BuildPresentationMatrix(&cb.mvp[0][0], p);
    D3D11_MAPPED_SUBRESOURCE cm = {};
    if (SUCCEEDED(p->context->Map(p->spriteCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &cm))) {
        memcpy(cm.pData, &cb, sizeof(cb));
        p->context->Unmap(p->spriteCB, 0);
    }

    UINT stride = sizeof(Model3DVertex), offset = 0;
    p->context->IASetVertexBuffers(0, 1, &p->model3DVB, &stride, &offset);
    p->context->IASetInputLayout(p->model3DLayout);
    p->context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    p->context->VSSetShader(p->model3DVS, nullptr, 0);
    p->context->VSSetConstantBuffers(0, 1, &p->spriteCB);
    p->context->PSSetShader(p->quadPS, nullptr, 0);

    ID3D11ShaderResourceView* srv = nullptr;
    if ((int)tex > 0 && (int)tex < MARNI_MAX_TEXTURES)
        srv = p->slots[tex].srv;
    if (!srv) srv = p->whiteSRV;
    if (!srv) return;
    p->context->PSSetShaderResources(0, 1, &srv);

    ID3D11SamplerState* s = (sampler == MARNI_SAMPLER_POINT) ? p->sampPoint : p->sampLinear;
    if (!s) s = p->sampLinear;
    p->context->PSSetSamplers(0, 1, &s);

    ID3D11BlendState* bs = p->blendAlpha;
    if (blend == MARNI_BLEND_ADD)       bs = p->blendAdd;
    else if (blend == MARNI_BLEND_DISABLE) bs = p->blendDisabled;
    if (!bs) bs = p->blendAlpha;
    float bf[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    p->context->OMSetBlendState(bs, bf, 0xFFFFFFFFu);

    // depthWrite=false: translucent geometry (water, glass - record +0x68
    // alpha < 1). Test against what is already in the buffer but never write:
    // the original's alpha-blended D3D7 primitives ran with ZWRITEOFF too.
    // Writing would hide the fade polys (ground shadows / blood pools) that
    // sit BEHIND such surfaces - e.g. every shadow under room40E0's water -
    // and would poison the buffer for the far-to-near painter walk around it.
    ID3D11DepthStencilState* ds = depthWrite ? p->depthEnabled : p->depthTestNoWrite;
    if (ds)
        p->context->OMSetDepthStencilState(ds, 0);

    p->context->Draw((UINT)(triCount * 3), 0);

    // Leave the pipeline in the 2D state the rest of the renderer expects.
    if (p->depthDisabled)
        p->context->OMSetDepthStencilState(p->depthDisabled, 0);
}

void MarniDX::DrawTrianglesPersp(const float* verts, int triCount, MarniHandle tex,
                                 MarniSampler sampler, MarniBlend blend,
                                 bool depthTest)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready || !verts || triCount <= 0) return;
    if (triCount > 1024) triCount = 1024;
    if (!p->context || !p->model3DVB || !p->spriteCB || !p->model3DVS
        || !p->quadPS || !p->model3DLayout) return;

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(p->context->Map(p->model3DVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
        return;
    memcpy(m.pData, verts, sizeof(Model3DVertex) * 3 * (size_t)triCount);
    p->context->Unmap(p->model3DVB, 0);

    SpriteConstantBuffer cb;
    BuildPresentationMatrix(&cb.mvp[0][0], p);
    D3D11_MAPPED_SUBRESOURCE cm = {};
    if (SUCCEEDED(p->context->Map(p->spriteCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &cm))) {
        memcpy(cm.pData, &cb, sizeof(cb));
        p->context->Unmap(p->spriteCB, 0);
    }

    UINT stride = sizeof(Model3DVertex), offset = 0;
    p->context->IASetVertexBuffers(0, 1, &p->model3DVB, &stride, &offset);
    p->context->IASetInputLayout(p->model3DLayout);
    p->context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    p->context->VSSetShader(p->model3DVS, nullptr, 0);
    p->context->VSSetConstantBuffers(0, 1, &p->spriteCB);
    p->context->PSSetShader(p->quadPS, nullptr, 0);

    ID3D11ShaderResourceView* srv = nullptr;
    if ((int)tex > 0 && (int)tex < MARNI_MAX_TEXTURES)
        srv = p->slots[tex].srv;
    if (!srv) srv = p->whiteSRV;
    if (!srv) return;
    p->context->PSSetShaderResources(0, 1, &srv);

    ID3D11SamplerState* s = (sampler == MARNI_SAMPLER_POINT) ? p->sampPoint : p->sampLinear;
    if (!s) s = p->sampLinear;
    p->context->PSSetSamplers(0, 1, &s);

    ID3D11BlendState* bs = p->blendAlpha;
    if (blend == MARNI_BLEND_ADD)       bs = p->blendAdd;
    else if (blend == MARNI_BLEND_DISABLE) bs = p->blendDisabled;
    if (!bs) bs = p->blendAlpha;
    float bf[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    p->context->OMSetBlendState(bs, bf, 0xFFFFFFFFu);

    // No depth by default: the shadow draws over the room like a sprite. This
    // used to inherit whatever the caller had set, which held only while the
    // whole 2D pass ran after the 3D one. FlushTmdObjects now interleaves
    // scene sprites between DrawTriangles3D batches, so state it outright
    // rather than depend on that call having restored it.
    //
    // depthTest=true instead clips against the model geometry like the
    // original's Z-buffered viewport quads (test only - never write). The
    // caller supplies real per-corner NDC z in that case.
    if (depthTest) {
        if (p->depthTestNoWrite)
            p->context->OMSetDepthStencilState(p->depthTestNoWrite, 0);
        p->context->Draw((UINT)(triCount * 3), 0);
        if (p->depthDisabled)
            p->context->OMSetDepthStencilState(p->depthDisabled, 0);
        return;
    }
    if (p->depthDisabled)
        p->context->OMSetDepthStencilState(p->depthDisabled, 0);

    p->context->Draw((UINT)(triCount * 3), 0);
}

void MarniDX::DrawRect(int x, int y, int w, int h, DWORD color)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready) return;
    float r = ((color >> 16) & 0xFF) / 255.0f;
    float g = ((color >> 8)  & 0xFF) / 255.0f;
    float b = ( color        & 0xFF) / 255.0f;
    float a = ((color >> 24) & 0xFF) / 255.0f;
    float x0 = (float)x, x1 = (float)(x + w);
    float y0 = (float)y, y1 = (float)(y + h);
    QuadVertex verts[6] = {
        { x0, y0, 0.0f, 0.0f, r, g, b, a },
        { x1, y0, 1.0f, 0.0f, r, g, b, a },
        { x0, y1, 0.0f, 1.0f, r, g, b, a },
        { x0, y1, 0.0f, 1.0f, r, g, b, a },
        { x1, y0, 1.0f, 0.0f, r, g, b, a },
        { x1, y1, 1.0f, 1.0f, r, g, b, a },
    };
    // rects use the white texture; encode via NULL handle -> white fallback.
    DrawQuadInternal(p, verts, MARNI_NULL_HANDLE,
                     MARNI_SAMPLER_POINT, MARNI_BLEND_ALPHA);
}

void MarniDX::DrawLine(float x0, float y0, float x1, float y1,
                       float thickness, DWORD color)
{
    Impl* p = m_pImpl;
    if (!p || !p->ready) return;

    float r = ((color >> 16) & 0xFF) / 255.0f;
    float g = ((color >> 8)  & 0xFF) / 255.0f;
    float b = ( color        & 0xFF) / 255.0f;
    float a = ((color >> 24) & 0xFF) / 255.0f;

    // Thicken the segment with a quad perpendicular to its direction.
    float dx = x1 - x0;
    float dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    float nx, ny;
    if (len < 0.001f) {
        nx = 1.0f;
        ny = 0.0f;
    } else {
        nx = -dy / len;
        ny =  dx / len;
    }
    float hw = thickness * 0.5f;
    float ox = nx * hw;
    float oy = ny * hw;

    QuadVertex verts[6] = {
        { x0 - ox, y0 - oy, 0.0f, 0.0f, r, g, b, a },
        { x1 - ox, y1 - oy, 1.0f, 0.0f, r, g, b, a },
        { x0 + ox, y0 + oy, 0.0f, 1.0f, r, g, b, a },
        { x0 + ox, y0 + oy, 0.0f, 1.0f, r, g, b, a },
        { x1 - ox, y1 - oy, 1.0f, 0.0f, r, g, b, a },
        { x1 + ox, y1 + oy, 1.0f, 1.0f, r, g, b, a },
    };
    DrawQuadInternal(p, verts, MARNI_NULL_HANDLE,
                     MARNI_SAMPLER_POINT, MARNI_BLEND_ALPHA);
}

// ============================================================================
// Readback
// ============================================================================
BOOL MarniDX::CaptureBackbufferToRGBA(void** outPixels, DWORD* outW, DWORD* outH)
{
    Impl* p = m_pImpl;
    if (!outPixels || !outW || !outH) return FALSE;
    *outPixels = nullptr; *outW = 0; *outH = 0;
    if (!p || !p->ready || !p->swapChain || !p->device || !p->context)
        return FALSE;

    ID3D11Texture2D* bb = nullptr;
    if (FAILED(p->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb)))
        return FALSE;

    D3D11_TEXTURE2D_DESC bd;
    bb->GetDesc(&bd);
    DWORD w = bd.Width, h = bd.Height;

    D3D11_TEXTURE2D_DESC sd = bd;
    sd.Usage          = D3D11_USAGE_STAGING;
    sd.BindFlags      = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags      = 0;
    ID3D11Texture2D* st = nullptr;
    if (FAILED(p->device->CreateTexture2D(&sd, nullptr, &st))) {
        bb->Release(); return FALSE;
    }
    p->context->CopyResource(st, bb);
    bb->Release();

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(p->context->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        st->Release(); return FALSE;
    }
    DWORD bufSize = w * h * 4;
    BYTE* rgba = (BYTE*)malloc(bufSize);
    if (!rgba) { p->context->Unmap(st, 0); st->Release(); return FALSE; }
    BYTE* dst = rgba; BYTE* src = (BYTE*)m.pData;
    for (DWORD y = 0; y < h; y++) {
        memcpy(dst, src, w * 4);
        dst += w * 4;
        src += m.RowPitch;
    }
    p->context->Unmap(st, 0);
    st->Release();
    *outPixels = rgba; *outW = w; *outH = h;
    return TRUE;
}

BOOL MarniDX::ReadTextureRGBA(MarniHandle tex, void** outPixels,
                              int* outWidth, int* outHeight)
{
    Impl* p = m_pImpl;
    if (!outPixels || !outWidth || !outHeight || !p || !p->ready)
        return FALSE;
    *outPixels = nullptr; *outWidth = 0; *outHeight = 0;
    if ((int)tex <= 0 || (int)tex >= MARNI_MAX_TEXTURES) return FALSE;
    ID3D11Texture2D* ttex = p->slots[tex].tex;
    if (!ttex) return FALSE;

    D3D11_TEXTURE2D_DESC d;
    ttex->GetDesc(&d);

    D3D11_TEXTURE2D_DESC sd = d;
    sd.Usage          = D3D11_USAGE_STAGING;
    sd.BindFlags      = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags      = 0;
    ID3D11Texture2D* st = nullptr;
    if (FAILED(p->device->CreateTexture2D(&sd, nullptr, &st))) return FALSE;
    p->context->CopyResource(st, ttex);
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(p->context->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        st->Release(); return FALSE;
    }
    int w = (int)d.Width, h = (int)d.Height;
    BYTE* rgba = (BYTE*)malloc((size_t)w * h * 4);
    if (!rgba) { p->context->Unmap(st, 0); st->Release(); return FALSE; }
    BYTE* dst = rgba; BYTE* src = (BYTE*)m.pData;
    for (int y = 0; y < h; y++) {
        memcpy(dst, src, (size_t)w * 4);
        dst += w * 4;
        src += m.RowPitch;
    }
    p->context->Unmap(st, 0);
    st->Release();
    *outPixels = rgba; *outWidth = w; *outHeight = h;
    return TRUE;
}

// ============================================================================
// Font texture accessor pair
// ============================================================================
void MarniDX::SetFontTexture(MarniHandle tex, int w, int h)
{
    Impl* p = m_pImpl;
    if (!p) return;
    p->fontHandle = tex;
    p->fontW = w;
    p->fontH = h;
}

MarniHandle MarniDX::FontTexture()    const { return m_pImpl ? m_pImpl->fontHandle : MARNI_NULL_HANDLE; }
int  MarniDX::FontTextureWidth()  const { return m_pImpl ? m_pImpl->fontW : 0; }
int  MarniDX::FontTextureHeight() const { return m_pImpl ? m_pImpl->fontH : 0; }

// Headless regression for the production GL shaders. Run check_depth_shader.py.
#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "depth_test_shaders.h"

static void require(bool ok, const char* message)
{
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static GLuint shader(GLenum kind, const char* source)
{
    GLuint result = glCreateShader(kind);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint ok = 0;
    glGetShaderiv(result, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetShaderInfoLog(result, sizeof(log), nullptr, log);
        std::fprintf(stderr, "%s\n", log);
    }
    require(ok, "Production shader compilation failed");
    return result;
}

struct Vertex { float x, y, u, v, r, g, b, a, depth, w; };

static void draw(int panels, bool inclined, bool blue = false)
{
    std::vector<Vertex> vertices;
    for (int panel = 0; panel < panels; ++panel) {
        const float x[] = {-1.0f + 2.0f*panel/panels,
                           -1.0f + 2.0f*(panel+1)/panels};
        const int corners[] = {0,1,2, 0,2,3};
        for (int corner : corners) {
            const float px = x[corner == 1 || corner == 2];
            const float t = (px+1.0f)*.5f;
            // Ray/plane intersection: inverse depth is affine in screen X.
            const float z = inclined ? 1.0f/((1-t)/1000.0f+t/9000.0f) : 2500.0f;
            vertices.push_back({px, corner < 2 ? -1.0f : 1.0f, 0, 0,
                               inclined && !blue ? 1.0f : 0.0f,
                               !inclined && !blue ? 1.0f : 0.0f,
                               blue ? 1.0f : 0.0f, 1,
                               (z-1.0f)/131071.0f, z});
        }
    }
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, vertices.size());
}

int main()
{
    auto platformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));
    require(platformDisplay != nullptr, "EGL platform extension unavailable");
    EGLDisplay display = platformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    require(eglInitialize(display, nullptr, nullptr), "Cannot initialize headless EGL");
    require(eglBindAPI(EGL_OPENGL_API), "Cannot select OpenGL");
    const EGLint attributes[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,EGL_OPENGL_BIT, EGL_RED_SIZE,8, EGL_GREEN_SIZE,8,
        EGL_BLUE_SIZE,8, EGL_DEPTH_SIZE,24, EGL_NONE};
    EGLConfig config; EGLint count;
    require(eglChooseConfig(display, attributes, &config, 1, &count) && count,
            "No EGL depth-buffer configuration");
    const EGLint surfaceAttributes[] = {EGL_WIDTH,64,EGL_HEIGHT,64,EGL_NONE};
    EGLSurface surface = eglCreatePbufferSurface(display, config, surfaceAttributes);
    const EGLint contextAttributes[] = {EGL_CONTEXT_MAJOR_VERSION,3,
        EGL_CONTEXT_MINOR_VERSION,3, EGL_CONTEXT_OPENGL_PROFILE_MASK,
        EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttributes);
    require(eglMakeCurrent(display, surface, surface, context), "Cannot activate EGL context");
    GLuint program = glCreateProgram();
    glAttachShader(program, shader(GL_VERTEX_SHADER, vertexSource));
    glAttachShader(program, shader(GL_FRAGMENT_SHADER, fragmentSource));
    glLinkProgram(program);
    GLint linked; glGetProgramiv(program, GL_LINK_STATUS, &linked);
    require(linked, "Production shader interface does not link");
    glUseProgram(program);
    const float identity[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    glUniformMatrix4fv(glGetUniformLocation(program,"uMVP"), 1, GL_FALSE, identity);
    glUniform1i(glGetUniformLocation(program,"uPersp"), 1);
    GLuint vao, buffer, texture;
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    glGenBuffers(1,&buffer); glBindBuffer(GL_ARRAY_BUFFER,buffer);
    const int sizes[] = {2,2,4,2}; const size_t offsets[] = {0,2,4,8};
    for (int i=0;i<4;++i) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i,sizes[i],GL_FLOAT,GL_FALSE,sizeof(Vertex),
                              reinterpret_cast<void*>(offsets[i]*sizeof(float)));
    }
    glGenTextures(1,&texture); glBindTexture(GL_TEXTURE_2D,texture);
    const unsigned char white[] = {255,255,255,255};
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,1,1,0,GL_RGBA,GL_UNSIGNED_BYTE,white);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glViewport(0,0,64,64); glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);
    const float t = 32.5f/64.0f;
    const float expected = (1.0f/((1-t)/1000+t/9000)-1)/131071;
    for (int panels : {1,8}) for (int reverse : {0,1}) {
        glClearDepth(1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        if (reverse) draw(panels,true);
        draw(1,false);
        if (!reverse) draw(panels,true);
        float depth; unsigned char color[4];
        glReadPixels(32,32,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&depth);
        glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,color);
        require(std::fabs(depth-expected)<0.000001f, "Inclined surface depth depends on triangulation");
        require(color[0]>250 && color[1]<5, "Object behind the plane incorrectly occludes it");
    }
    glDisable(GL_DEPTH_TEST);
    glUniform1i(glGetUniformLocation(program,"uPersp"), 0);
    draw(1,false,true);
    unsigned char overlay[4]; glReadPixels(32,32,1,1,GL_RGBA,GL_UNSIGNED_BYTE,overlay);
    require(overlay[2]>250 && overlay[0]<5, "2D overlay failed");
    require(glGetError()==GL_NO_ERROR, "OpenGL error during regression");
    std::puts("Production GL depth: planar intersection, subdivision, draw order and overlay passed");
    eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
    eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
}

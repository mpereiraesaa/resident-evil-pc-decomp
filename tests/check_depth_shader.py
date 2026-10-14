#!/usr/bin/env python3
"""Run production GL shaders in headless EGL. Requires g++, pkg-config, EGL/GL dev libraries.

The harness is host-native; it tests actual shaders, not the 32-bit game ABI.
Use --affine-control to confirm the old depth behavior fails this regression.
"""
import argparse
import ast
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--affine-control', action='store_true')
    args = parser.parse_args()
    source = (ROOT / 'src/marni/MarniDX_GL.cpp').read_text()
    shaders = []
    for name in ('kVertexSrc', 'kFragmentSrc'):
        start = source.index('static const char* ' + name + ' =')
        end = source.index('\n\n', start)
        strings = re.findall(r'"(?:[^"\\]|\\.)*"', source[start:end])
        shaders.append(''.join(ast.literal_eval(token) for token in strings))
    if args.affine_control:
        if 'gl_FragDepth = vDepth;' not in shaders[1]:
            raise RuntimeError('Production depth assignment changed; review negative control')
        shaders[1] = shaders[1].replace('gl_FragDepth = vDepth;',
                                      'gl_FragDepth = gl_FragCoord.z;')
    with tempfile.TemporaryDirectory(prefix='re1-depth-test-') as directory:
        build = Path(directory)
        header = '\n'.join('static const char* ' + name + ' = R"SHADER(' + shader + ')SHADER";'
                           for name, shader in zip(('vertexSource', 'fragmentSource'), shaders))
        (build / 'depth_test_shaders.h').write_text(header)
        flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', 'egl', 'gl'],
                                        text=True).split()
        executable = build / 'depth-test'
        subprocess.run(['g++', '-std=c++17', '-I'+directory,
                        str(ROOT / 'tests/depth_shader_test.cpp'), '-o', str(executable),
                        *flags], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    main()

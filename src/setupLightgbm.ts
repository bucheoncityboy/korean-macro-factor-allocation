import { execFileSync } from 'node:child_process';
import { access, mkdir } from 'node:fs/promises';
const path = '.cache/LightGBM';
await mkdir('.cache', { recursive: true });
try { await access(`${path}/.git`); } catch {
  execFileSync('git', ['clone', '--depth', '1', '--branch', 'v4.6.0', '--recurse-submodules', '--shallow-submodules',
    'https://github.com/microsoft/LightGBM.git', path], { stdio: 'inherit' });
}
execFileSync('cmake', ['-B', `${path}/build`, '-S', path, '-DUSE_OPENMP=OFF', '-DBUILD_CLI=ON', '-DINSTALL_HEADERS=OFF', '-DCMAKE_BUILD_TYPE=Release'], { stdio: 'inherit' });
execFileSync('cmake', ['--build', `${path}/build`, '-j4'], { stdio: 'inherit' });
console.log('Native LightGBM 4.6.0 ready at .cache/LightGBM/lightgbm');

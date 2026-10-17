import test from 'node:test';
import assert from 'node:assert/strict';
import { designSensitivity, assertPaired } from '../src/regimes/power.js';
import { months } from '../src/regimes/data.js';

test('fixed-effect power has corrected null size and N scales inversely with squared effect',()=>{
  const a=designSensitivity(.01,200,.02),b=designSensitivity(.01,200,.04);
  assert.ok(Math.abs(a.approximate_required_total_n-4*b.approximate_required_total_n)<=4);
  const nearNull=designSensitivity(.01,200,1e-12);
  assert.ok(Math.abs(nearNull.approximate_power_at_fixed_effect-.05/12)<1e-8);
  const boundary=designSensitivity(.01,200,a.approximate_80pct_MDE);
  assert.ok(Math.abs(boundary.approximate_power_at_fixed_effect-.8)<1e-7);
  assert.ok(designSensitivity(.01,200,.02,12).approximate_80pct_MDE>designSensitivity(.01,200,.02,6).approximate_80pct_MDE);
});
test('loss comparison refuses missing, reordered and duplicated months',()=>{
  const a=[{period:'2000-01'},{period:'2000-02'}];assertPaired(a,[...a]);
  assert.throws(()=>assertPaired(a,a.slice(1)));assert.throws(()=>assertPaired(a,[...a].reverse()));assert.throws(()=>assertPaired([a[0],a[0]],[a[0],a[0]]));
});
test('expanded real-data masks add 36 forecast months and 25 ETF intervals',()=>{
  assert.equal(months('2006-09','2026-05').length,237);assert.equal(months('2014-10','2026-05').length,140);
});

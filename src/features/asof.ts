export const monthIndex = (m: string) => Number(m.slice(0, 4)) * 12 + Number(m.slice(5, 7)) - 1;
export function addMonths(month: string, n: number) {
  const i = monthIndex(month) + n;
  return `${Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, '0')}`;
}

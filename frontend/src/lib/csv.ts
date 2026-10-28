/** Minimal RFC 4180 CSV parser (quoted fields, doubled quotes, CRLF). Returns one object per data row. */
export function parseCsv(src: string): Record<string, string>[] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < src.length; i++) {
    const c = src[i]
    if (quoted) {
      if (c === '"' && src[i + 1] === '"') {
        field += '"'
        i++
      } else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') {
      row.push(field)
      field = ''
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && src[i + 1] === '\n') i++
      row.push(field)
      field = ''
      if (row.some((f) => f !== '')) rows.push(row)
      row = []
    } else field += c
  }
  if (field || row.length) {
    row.push(field)
    if (row.some((f) => f !== '')) rows.push(row)
  }
  const [head = [], ...body] = rows
  const keys = head.map((h) => h.trim().replace(/^\uFEFF/, ''))
  return body.map((r) => Object.fromEntries(keys.map((k, i) => [k, r[i] ?? ''])))
}

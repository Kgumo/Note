import {
  __name
} from "./chunk-H42KCMGY.js";

// node_modules/mermaid/dist/chunks/mermaid.core/chunk-353BL4L5.mjs
function populateCommonDb(ast, db) {
  if (ast.accDescr) {
    db.setAccDescription?.(ast.accDescr);
  }
  if (ast.accTitle) {
    db.setAccTitle?.(ast.accTitle);
  }
  if (ast.title) {
    db.setDiagramTitle?.(ast.title);
  }
}
__name(populateCommonDb, "populateCommonDb");

export {
  populateCommonDb
};
//# sourceMappingURL=chunk-PQCOBJL7.js.map

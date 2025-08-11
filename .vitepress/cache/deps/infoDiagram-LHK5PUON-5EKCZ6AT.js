import {
  parse
} from "./chunk-RBF2APTZ.js";
import "./chunk-BFIQR5IZ.js";
import "./chunk-WXMEHLOL.js";
import "./chunk-UHGULKYM.js";
import "./chunk-WQSQIAVS.js";
import "./chunk-QEKKNWQV.js";
import "./chunk-EPVJS4TN.js";
import "./chunk-OSXRMYXV.js";
import "./chunk-WR4DO2U2.js";
import "./chunk-3KOL2IQZ.js";
import {
  package_default
} from "./chunk-MJO4EG7R.js";
import {
  selectSvgElement
} from "./chunk-KAQLTNHE.js";
import "./chunk-NBWFZMTS.js";
import {
  __name,
  configureSvgSize,
  log
} from "./chunk-H42KCMGY.js";
import "./chunk-ST3SR5TB.js";
import "./chunk-JJGIA2MQ.js";
import "./chunk-IKZWERSR.js";

// node_modules/mermaid/dist/chunks/mermaid.core/infoDiagram-LHK5PUON.mjs
var parser = {
  parse: __name(async (input) => {
    const ast = await parse("info", input);
    log.debug(ast);
  }, "parse")
};
var DEFAULT_INFO_DB = {
  version: package_default.version + (true ? "" : "-tiny")
};
var getVersion = __name(() => DEFAULT_INFO_DB.version, "getVersion");
var db = {
  getVersion
};
var draw = __name((text, id, version) => {
  log.debug("rendering info diagram\n" + text);
  const svg = selectSvgElement(id);
  configureSvgSize(svg, 100, 400, true);
  const group = svg.append("g");
  group.append("text").attr("x", 100).attr("y", 40).attr("class", "version").attr("font-size", 32).style("text-anchor", "middle").text(`v${version}`);
}, "draw");
var renderer = { draw };
var diagram = {
  parser,
  db,
  renderer
};
export {
  diagram
};
//# sourceMappingURL=infoDiagram-LHK5PUON-5EKCZ6AT.js.map

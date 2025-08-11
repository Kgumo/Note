import {
  __name
} from "./chunk-H42KCMGY.js";

// node_modules/mermaid/dist/chunks/mermaid.core/chunk-AACKK3MU.mjs
var ImperativeState = class {
  /**
   * @param init - Function that creates the default state.
   */
  constructor(init) {
    this.init = init;
    this.records = this.init();
  }
  static {
    __name(this, "ImperativeState");
  }
  reset() {
    this.records = this.init();
  }
};

export {
  ImperativeState
};
//# sourceMappingURL=chunk-WPS4IINC.js.map

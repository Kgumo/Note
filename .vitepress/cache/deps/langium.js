import {
  __commonJS,
  __export,
  __reExport,
  __toESM
} from "./chunk-IKZWERSR.js";

// node_modules/vscode-jsonrpc/lib/common/ral.js
var require_ral = __commonJS({
  "node_modules/vscode-jsonrpc/lib/common/ral.js"(exports2) {
    "use strict";
    Object.defineProperty(exports2, "__esModule", { value: true });
    var _ral;
    function RAL() {
      if (_ral === void 0) {
        throw new Error(`No runtime abstraction layer installed`);
      }
      return _ral;
    }
    (function(RAL2) {
      function install(ral) {
        if (ral === void 0) {
          throw new Error(`No runtime abstraction layer provided`);
        }
        _ral = ral;
      }
      RAL2.install = install;
    })(RAL || (RAL = {}));
    exports2.default = RAL;
  }
});

// node_modules/vscode-jsonrpc/lib/common/is.js
var require_is = __commonJS({
  "node_modules/vscode-jsonrpc/lib/common/is.js"(exports2) {
    "use strict";
    Object.defineProperty(exports2, "__esModule", { value: true });
    exports2.stringArray = exports2.array = exports2.func = exports2.error = exports2.number = exports2.string = exports2.boolean = void 0;
    function boolean(value) {
      return value === true || value === false;
    }
    exports2.boolean = boolean;
    function string(value) {
      return typeof value === "string" || value instanceof String;
    }
    exports2.string = string;
    function number(value) {
      return typeof value === "number" || value instanceof Number;
    }
    exports2.number = number;
    function error(value) {
      return value instanceof Error;
    }
    exports2.error = error;
    function func(value) {
      return typeof value === "function";
    }
    exports2.func = func;
    function array(value) {
      return Array.isArray(value);
    }
    exports2.array = array;
    function stringArray(value) {
      return array(value) && value.every((elem) => string(elem));
    }
    exports2.stringArray = stringArray;
  }
});

// node_modules/vscode-jsonrpc/lib/common/events.js
var require_events = __commonJS({
  "node_modules/vscode-jsonrpc/lib/common/events.js"(exports2) {
    "use strict";
    Object.defineProperty(exports2, "__esModule", { value: true });
    exports2.Emitter = exports2.Event = void 0;
    var ral_1 = require_ral();
    var Event;
    (function(Event2) {
      const _disposable = { dispose() {
      } };
      Event2.None = function() {
        return _disposable;
      };
    })(Event || (exports2.Event = Event = {}));
    var CallbackList = class {
      add(callback, context = null, bucket) {
        if (!this._callbacks) {
          this._callbacks = [];
          this._contexts = [];
        }
        this._callbacks.push(callback);
        this._contexts.push(context);
        if (Array.isArray(bucket)) {
          bucket.push({ dispose: () => this.remove(callback, context) });
        }
      }
      remove(callback, context = null) {
        if (!this._callbacks) {
          return;
        }
        let foundCallbackWithDifferentContext = false;
        for (let i = 0, len = this._callbacks.length; i < len; i++) {
          if (this._callbacks[i] === callback) {
            if (this._contexts[i] === context) {
              this._callbacks.splice(i, 1);
              this._contexts.splice(i, 1);
              return;
            } else {
              foundCallbackWithDifferentContext = true;
            }
          }
        }
        if (foundCallbackWithDifferentContext) {
          throw new Error("When adding a listener with a context, you should remove it with the same context");
        }
      }
      invoke(...args) {
        if (!this._callbacks) {
          return [];
        }
        const ret = [], callbacks = this._callbacks.slice(0), contexts = this._contexts.slice(0);
        for (let i = 0, len = callbacks.length; i < len; i++) {
          try {
            ret.push(callbacks[i].apply(contexts[i], args));
          } catch (e) {
            (0, ral_1.default)().console.error(e);
          }
        }
        return ret;
      }
      isEmpty() {
        return !this._callbacks || this._callbacks.length === 0;
      }
      dispose() {
        this._callbacks = void 0;
        this._contexts = void 0;
      }
    };
    var Emitter3 = class _Emitter {
      constructor(_options) {
        this._options = _options;
      }
      /**
       * For the public to allow to subscribe
       * to events from this Emitter
       */
      get event() {
        if (!this._event) {
          this._event = (listener, thisArgs, disposables) => {
            if (!this._callbacks) {
              this._callbacks = new CallbackList();
            }
            if (this._options && this._options.onFirstListenerAdd && this._callbacks.isEmpty()) {
              this._options.onFirstListenerAdd(this);
            }
            this._callbacks.add(listener, thisArgs);
            const result2 = {
              dispose: () => {
                if (!this._callbacks) {
                  return;
                }
                this._callbacks.remove(listener, thisArgs);
                result2.dispose = _Emitter._noop;
                if (this._options && this._options.onLastListenerRemove && this._callbacks.isEmpty()) {
                  this._options.onLastListenerRemove(this);
                }
              }
            };
            if (Array.isArray(disposables)) {
              disposables.push(result2);
            }
            return result2;
          };
        }
        return this._event;
      }
      /**
       * To be kept private to fire an event to
       * subscribers
       */
      fire(event) {
        if (this._callbacks) {
          this._callbacks.invoke.call(this._callbacks, event);
        }
      }
      dispose() {
        if (this._callbacks) {
          this._callbacks.dispose();
          this._callbacks = void 0;
        }
      }
    };
    exports2.Emitter = Emitter3;
    Emitter3._noop = function() {
    };
  }
});

// node_modules/vscode-jsonrpc/lib/common/cancellation.js
var require_cancellation = __commonJS({
  "node_modules/vscode-jsonrpc/lib/common/cancellation.js"(exports2) {
    "use strict";
    Object.defineProperty(exports2, "__esModule", { value: true });
    exports2.CancellationTokenSource = exports2.CancellationToken = void 0;
    var ral_1 = require_ral();
    var Is2 = require_is();
    var events_1 = require_events();
    var CancellationToken11;
    (function(CancellationToken12) {
      CancellationToken12.None = Object.freeze({
        isCancellationRequested: false,
        onCancellationRequested: events_1.Event.None
      });
      CancellationToken12.Cancelled = Object.freeze({
        isCancellationRequested: true,
        onCancellationRequested: events_1.Event.None
      });
      function is(value) {
        const candidate = value;
        return candidate && (candidate === CancellationToken12.None || candidate === CancellationToken12.Cancelled || Is2.boolean(candidate.isCancellationRequested) && !!candidate.onCancellationRequested);
      }
      CancellationToken12.is = is;
    })(CancellationToken11 || (exports2.CancellationToken = CancellationToken11 = {}));
    var shortcutEvent = Object.freeze(function(callback, context) {
      const handle = (0, ral_1.default)().timer.setTimeout(callback.bind(context), 0);
      return { dispose() {
        handle.dispose();
      } };
    });
    var MutableToken = class {
      constructor() {
        this._isCancelled = false;
      }
      cancel() {
        if (!this._isCancelled) {
          this._isCancelled = true;
          if (this._emitter) {
            this._emitter.fire(void 0);
            this.dispose();
          }
        }
      }
      get isCancellationRequested() {
        return this._isCancelled;
      }
      get onCancellationRequested() {
        if (this._isCancelled) {
          return shortcutEvent;
        }
        if (!this._emitter) {
          this._emitter = new events_1.Emitter();
        }
        return this._emitter.event;
      }
      dispose() {
        if (this._emitter) {
          this._emitter.dispose();
          this._emitter = void 0;
        }
      }
    };
    var CancellationTokenSource3 = class {
      get token() {
        if (!this._token) {
          this._token = new MutableToken();
        }
        return this._token;
      }
      cancel() {
        if (!this._token) {
          this._token = CancellationToken11.Cancelled;
        } else {
          this._token.cancel();
        }
      }
      dispose() {
        if (!this._token) {
          this._token = CancellationToken11.None;
        } else if (this._token instanceof MutableToken) {
          this._token.dispose();
        }
      }
    };
    exports2.CancellationTokenSource = CancellationTokenSource3;
  }
});

// node_modules/langium/lib/index.js
var lib_exports = {};
__export(lib_exports, {
  AbstractAstReflection: () => AbstractAstReflection,
  AbstractCstNode: () => AbstractCstNode,
  AbstractLangiumParser: () => AbstractLangiumParser,
  AbstractParserErrorMessageProvider: () => AbstractParserErrorMessageProvider,
  AbstractThreadedAsyncParser: () => AbstractThreadedAsyncParser,
  AstUtils: () => ast_utils_exports,
  BiMap: () => BiMap,
  Cancellation: () => cancellation_exports,
  CompositeCstNodeImpl: () => CompositeCstNodeImpl,
  ContextCache: () => ContextCache,
  CstNodeBuilder: () => CstNodeBuilder,
  CstUtils: () => cst_utils_exports,
  DEFAULT_TOKENIZE_OPTIONS: () => DEFAULT_TOKENIZE_OPTIONS,
  DONE_RESULT: () => DONE_RESULT,
  DatatypeSymbol: () => DatatypeSymbol,
  DefaultAstNodeDescriptionProvider: () => DefaultAstNodeDescriptionProvider,
  DefaultAstNodeLocator: () => DefaultAstNodeLocator,
  DefaultAsyncParser: () => DefaultAsyncParser,
  DefaultCommentProvider: () => DefaultCommentProvider,
  DefaultConfigurationProvider: () => DefaultConfigurationProvider,
  DefaultDocumentBuilder: () => DefaultDocumentBuilder,
  DefaultDocumentValidator: () => DefaultDocumentValidator,
  DefaultHydrator: () => DefaultHydrator,
  DefaultIndexManager: () => DefaultIndexManager,
  DefaultJsonSerializer: () => DefaultJsonSerializer,
  DefaultLangiumDocumentFactory: () => DefaultLangiumDocumentFactory,
  DefaultLangiumDocuments: () => DefaultLangiumDocuments,
  DefaultLexer: () => DefaultLexer,
  DefaultLexerErrorMessageProvider: () => DefaultLexerErrorMessageProvider,
  DefaultLinker: () => DefaultLinker,
  DefaultNameProvider: () => DefaultNameProvider,
  DefaultReferenceDescriptionProvider: () => DefaultReferenceDescriptionProvider,
  DefaultReferences: () => DefaultReferences,
  DefaultScopeComputation: () => DefaultScopeComputation,
  DefaultScopeProvider: () => DefaultScopeProvider,
  DefaultServiceRegistry: () => DefaultServiceRegistry,
  DefaultTokenBuilder: () => DefaultTokenBuilder,
  DefaultValueConverter: () => DefaultValueConverter,
  DefaultWorkspaceLock: () => DefaultWorkspaceLock,
  DefaultWorkspaceManager: () => DefaultWorkspaceManager,
  Deferred: () => Deferred,
  Disposable: () => Disposable,
  DisposableCache: () => DisposableCache,
  DocumentCache: () => DocumentCache,
  DocumentState: () => DocumentState,
  DocumentValidator: () => DocumentValidator,
  EMPTY_SCOPE: () => EMPTY_SCOPE,
  EMPTY_STREAM: () => EMPTY_STREAM,
  EmptyFileSystem: () => EmptyFileSystem,
  EmptyFileSystemProvider: () => EmptyFileSystemProvider,
  ErrorWithLocation: () => ErrorWithLocation,
  GrammarAST: () => ast_exports,
  GrammarUtils: () => grammar_utils_exports,
  IndentationAwareLexer: () => IndentationAwareLexer,
  IndentationAwareTokenBuilder: () => IndentationAwareTokenBuilder,
  JSDocDocumentationProvider: () => JSDocDocumentationProvider,
  LangiumCompletionParser: () => LangiumCompletionParser,
  LangiumParser: () => LangiumParser,
  LangiumParserErrorMessageProvider: () => LangiumParserErrorMessageProvider,
  LeafCstNodeImpl: () => LeafCstNodeImpl,
  LexingMode: () => LexingMode,
  MapScope: () => MapScope,
  Module: () => Module,
  MultiMap: () => MultiMap,
  OperationCancelled: () => OperationCancelled,
  ParserWorker: () => ParserWorker,
  Reduction: () => Reduction,
  RegExpUtils: () => regexp_utils_exports,
  RootCstNodeImpl: () => RootCstNodeImpl,
  SimpleCache: () => SimpleCache,
  StreamImpl: () => StreamImpl,
  StreamScope: () => StreamScope,
  TextDocument: () => TextDocument2,
  TreeStreamImpl: () => TreeStreamImpl,
  URI: () => URI2,
  UriUtils: () => UriUtils,
  ValidationCategory: () => ValidationCategory,
  ValidationRegistry: () => ValidationRegistry,
  ValueConverter: () => ValueConverter,
  WorkspaceCache: () => WorkspaceCache,
  assertUnreachable: () => assertUnreachable,
  createCompletionParser: () => createCompletionParser,
  createDefaultCoreModule: () => createDefaultCoreModule,
  createDefaultSharedCoreModule: () => createDefaultSharedCoreModule,
  createGrammarConfig: () => createGrammarConfig,
  createLangiumParser: () => createLangiumParser,
  createParser: () => createParser,
  delayNextTick: () => delayNextTick,
  diagnosticData: () => diagnosticData,
  eagerLoad: () => eagerLoad,
  getDiagnosticRange: () => getDiagnosticRange,
  indentationBuilderDefaultOptions: () => indentationBuilderDefaultOptions,
  inject: () => inject,
  interruptAndCheck: () => interruptAndCheck,
  isAstNode: () => isAstNode,
  isAstNodeDescription: () => isAstNodeDescription,
  isAstNodeWithComment: () => isAstNodeWithComment,
  isCompositeCstNode: () => isCompositeCstNode,
  isIMultiModeLexerDefinition: () => isIMultiModeLexerDefinition,
  isJSDoc: () => isJSDoc,
  isLeafCstNode: () => isLeafCstNode,
  isLinkingError: () => isLinkingError,
  isNamed: () => isNamed,
  isOperationCancelled: () => isOperationCancelled,
  isReference: () => isReference,
  isRootCstNode: () => isRootCstNode,
  isTokenTypeArray: () => isTokenTypeArray,
  isTokenTypeDictionary: () => isTokenTypeDictionary,
  loadGrammarFromJson: () => loadGrammarFromJson,
  parseJSDoc: () => parseJSDoc,
  prepareLangiumParser: () => prepareLangiumParser,
  setInterruptionPeriod: () => setInterruptionPeriod,
  startCancelableOperation: () => startCancelableOperation,
  stream: () => stream,
  toDiagnosticData: () => toDiagnosticData,
  toDiagnosticSeverity: () => toDiagnosticSeverity
});

// node_modules/langium/lib/utils/cst-utils.js
var cst_utils_exports = {};
__export(cst_utils_exports, {
  DefaultNameRegexp: () => DefaultNameRegexp,
  RangeComparison: () => RangeComparison,
  compareRange: () => compareRange,
  findCommentNode: () => findCommentNode,
  findDeclarationNodeAtOffset: () => findDeclarationNodeAtOffset,
  findLeafNodeAtOffset: () => findLeafNodeAtOffset,
  findLeafNodeBeforeOffset: () => findLeafNodeBeforeOffset,
  flattenCst: () => flattenCst,
  getInteriorNodes: () => getInteriorNodes,
  getNextNode: () => getNextNode,
  getPreviousNode: () => getPreviousNode,
  getStartlineNode: () => getStartlineNode,
  inRange: () => inRange,
  isChildNode: () => isChildNode,
  isCommentNode: () => isCommentNode,
  streamCst: () => streamCst,
  toDocumentSegment: () => toDocumentSegment,
  tokenToRange: () => tokenToRange
});

// node_modules/langium/lib/syntax-tree.js
function isAstNode(obj) {
  return typeof obj === "object" && obj !== null && typeof obj.$type === "string";
}
function isReference(obj) {
  return typeof obj === "object" && obj !== null && typeof obj.$refText === "string";
}
function isAstNodeDescription(obj) {
  return typeof obj === "object" && obj !== null && typeof obj.name === "string" && typeof obj.type === "string" && typeof obj.path === "string";
}
function isLinkingError(obj) {
  return typeof obj === "object" && obj !== null && isAstNode(obj.container) && isReference(obj.reference) && typeof obj.message === "string";
}
var AbstractAstReflection = class {
  constructor() {
    this.subtypes = {};
    this.allSubtypes = {};
  }
  isInstance(node, type) {
    return isAstNode(node) && this.isSubtype(node.$type, type);
  }
  isSubtype(subtype, supertype) {
    if (subtype === supertype) {
      return true;
    }
    let nested = this.subtypes[subtype];
    if (!nested) {
      nested = this.subtypes[subtype] = {};
    }
    const existing = nested[supertype];
    if (existing !== void 0) {
      return existing;
    } else {
      const result2 = this.computeIsSubtype(subtype, supertype);
      nested[supertype] = result2;
      return result2;
    }
  }
  getAllSubTypes(type) {
    const existing = this.allSubtypes[type];
    if (existing) {
      return existing;
    } else {
      const allTypes = this.getAllTypes();
      const types = [];
      for (const possibleSubType of allTypes) {
        if (this.isSubtype(possibleSubType, type)) {
          types.push(possibleSubType);
        }
      }
      this.allSubtypes[type] = types;
      return types;
    }
  }
};
function isCompositeCstNode(node) {
  return typeof node === "object" && node !== null && Array.isArray(node.content);
}
function isLeafCstNode(node) {
  return typeof node === "object" && node !== null && typeof node.tokenType === "object";
}
function isRootCstNode(node) {
  return isCompositeCstNode(node) && typeof node.fullText === "string";
}

// node_modules/langium/lib/utils/stream.js
var StreamImpl = class _StreamImpl {
  constructor(startFn, nextFn) {
    this.startFn = startFn;
    this.nextFn = nextFn;
  }
  iterator() {
    const iterator = {
      state: this.startFn(),
      next: () => this.nextFn(iterator.state),
      [Symbol.iterator]: () => iterator
    };
    return iterator;
  }
  [Symbol.iterator]() {
    return this.iterator();
  }
  isEmpty() {
    const iterator = this.iterator();
    return Boolean(iterator.next().done);
  }
  count() {
    const iterator = this.iterator();
    let count = 0;
    let next = iterator.next();
    while (!next.done) {
      count++;
      next = iterator.next();
    }
    return count;
  }
  toArray() {
    const result2 = [];
    const iterator = this.iterator();
    let next;
    do {
      next = iterator.next();
      if (next.value !== void 0) {
        result2.push(next.value);
      }
    } while (!next.done);
    return result2;
  }
  toSet() {
    return new Set(this);
  }
  toMap(keyFn, valueFn) {
    const entryStream = this.map((element) => [
      keyFn ? keyFn(element) : element,
      valueFn ? valueFn(element) : element
    ]);
    return new Map(entryStream);
  }
  toString() {
    return this.join();
  }
  concat(other) {
    return new _StreamImpl(() => ({ first: this.startFn(), firstDone: false, iterator: other[Symbol.iterator]() }), (state) => {
      let result2;
      if (!state.firstDone) {
        do {
          result2 = this.nextFn(state.first);
          if (!result2.done) {
            return result2;
          }
        } while (!result2.done);
        state.firstDone = true;
      }
      do {
        result2 = state.iterator.next();
        if (!result2.done) {
          return result2;
        }
      } while (!result2.done);
      return DONE_RESULT;
    });
  }
  join(separator = ",") {
    const iterator = this.iterator();
    let value = "";
    let result2;
    let addSeparator = false;
    do {
      result2 = iterator.next();
      if (!result2.done) {
        if (addSeparator) {
          value += separator;
        }
        value += toString(result2.value);
      }
      addSeparator = true;
    } while (!result2.done);
    return value;
  }
  indexOf(searchElement, fromIndex = 0) {
    const iterator = this.iterator();
    let index = 0;
    let next = iterator.next();
    while (!next.done) {
      if (index >= fromIndex && next.value === searchElement) {
        return index;
      }
      next = iterator.next();
      index++;
    }
    return -1;
  }
  every(predicate) {
    const iterator = this.iterator();
    let next = iterator.next();
    while (!next.done) {
      if (!predicate(next.value)) {
        return false;
      }
      next = iterator.next();
    }
    return true;
  }
  some(predicate) {
    const iterator = this.iterator();
    let next = iterator.next();
    while (!next.done) {
      if (predicate(next.value)) {
        return true;
      }
      next = iterator.next();
    }
    return false;
  }
  forEach(callbackfn) {
    const iterator = this.iterator();
    let index = 0;
    let next = iterator.next();
    while (!next.done) {
      callbackfn(next.value, index);
      next = iterator.next();
      index++;
    }
  }
  map(callbackfn) {
    return new _StreamImpl(this.startFn, (state) => {
      const { done, value } = this.nextFn(state);
      if (done) {
        return DONE_RESULT;
      } else {
        return { done: false, value: callbackfn(value) };
      }
    });
  }
  filter(predicate) {
    return new _StreamImpl(this.startFn, (state) => {
      let result2;
      do {
        result2 = this.nextFn(state);
        if (!result2.done && predicate(result2.value)) {
          return result2;
        }
      } while (!result2.done);
      return DONE_RESULT;
    });
  }
  nonNullable() {
    return this.filter((e) => e !== void 0 && e !== null);
  }
  reduce(callbackfn, initialValue) {
    const iterator = this.iterator();
    let previousValue = initialValue;
    let next = iterator.next();
    while (!next.done) {
      if (previousValue === void 0) {
        previousValue = next.value;
      } else {
        previousValue = callbackfn(previousValue, next.value);
      }
      next = iterator.next();
    }
    return previousValue;
  }
  reduceRight(callbackfn, initialValue) {
    return this.recursiveReduce(this.iterator(), callbackfn, initialValue);
  }
  recursiveReduce(iterator, callbackfn, initialValue) {
    const next = iterator.next();
    if (next.done) {
      return initialValue;
    }
    const previousValue = this.recursiveReduce(iterator, callbackfn, initialValue);
    if (previousValue === void 0) {
      return next.value;
    }
    return callbackfn(previousValue, next.value);
  }
  find(predicate) {
    const iterator = this.iterator();
    let next = iterator.next();
    while (!next.done) {
      if (predicate(next.value)) {
        return next.value;
      }
      next = iterator.next();
    }
    return void 0;
  }
  findIndex(predicate) {
    const iterator = this.iterator();
    let index = 0;
    let next = iterator.next();
    while (!next.done) {
      if (predicate(next.value)) {
        return index;
      }
      next = iterator.next();
      index++;
    }
    return -1;
  }
  includes(searchElement) {
    const iterator = this.iterator();
    let next = iterator.next();
    while (!next.done) {
      if (next.value === searchElement) {
        return true;
      }
      next = iterator.next();
    }
    return false;
  }
  flatMap(callbackfn) {
    return new _StreamImpl(() => ({ this: this.startFn() }), (state) => {
      do {
        if (state.iterator) {
          const next = state.iterator.next();
          if (next.done) {
            state.iterator = void 0;
          } else {
            return next;
          }
        }
        const { done, value } = this.nextFn(state.this);
        if (!done) {
          const mapped = callbackfn(value);
          if (isIterable(mapped)) {
            state.iterator = mapped[Symbol.iterator]();
          } else {
            return { done: false, value: mapped };
          }
        }
      } while (state.iterator);
      return DONE_RESULT;
    });
  }
  flat(depth) {
    if (depth === void 0) {
      depth = 1;
    }
    if (depth <= 0) {
      return this;
    }
    const stream2 = depth > 1 ? this.flat(depth - 1) : this;
    return new _StreamImpl(() => ({ this: stream2.startFn() }), (state) => {
      do {
        if (state.iterator) {
          const next = state.iterator.next();
          if (next.done) {
            state.iterator = void 0;
          } else {
            return next;
          }
        }
        const { done, value } = stream2.nextFn(state.this);
        if (!done) {
          if (isIterable(value)) {
            state.iterator = value[Symbol.iterator]();
          } else {
            return { done: false, value };
          }
        }
      } while (state.iterator);
      return DONE_RESULT;
    });
  }
  head() {
    const iterator = this.iterator();
    const result2 = iterator.next();
    if (result2.done) {
      return void 0;
    }
    return result2.value;
  }
  tail(skipCount = 1) {
    return new _StreamImpl(() => {
      const state = this.startFn();
      for (let i = 0; i < skipCount; i++) {
        const next = this.nextFn(state);
        if (next.done) {
          return state;
        }
      }
      return state;
    }, this.nextFn);
  }
  limit(maxSize) {
    return new _StreamImpl(() => ({ size: 0, state: this.startFn() }), (state) => {
      state.size++;
      if (state.size > maxSize) {
        return DONE_RESULT;
      }
      return this.nextFn(state.state);
    });
  }
  distinct(by) {
    return new _StreamImpl(() => ({ set: /* @__PURE__ */ new Set(), internalState: this.startFn() }), (state) => {
      let result2;
      do {
        result2 = this.nextFn(state.internalState);
        if (!result2.done) {
          const value = by ? by(result2.value) : result2.value;
          if (!state.set.has(value)) {
            state.set.add(value);
            return result2;
          }
        }
      } while (!result2.done);
      return DONE_RESULT;
    });
  }
  exclude(other, key) {
    const otherKeySet = /* @__PURE__ */ new Set();
    for (const item of other) {
      const value = key ? key(item) : item;
      otherKeySet.add(value);
    }
    return this.filter((e) => {
      const ownKey = key ? key(e) : e;
      return !otherKeySet.has(ownKey);
    });
  }
};
function toString(item) {
  if (typeof item === "string") {
    return item;
  }
  if (typeof item === "undefined") {
    return "undefined";
  }
  if (typeof item.toString === "function") {
    return item.toString();
  }
  return Object.prototype.toString.call(item);
}
function isIterable(obj) {
  return !!obj && typeof obj[Symbol.iterator] === "function";
}
var EMPTY_STREAM = new StreamImpl(() => void 0, () => DONE_RESULT);
var DONE_RESULT = Object.freeze({ done: true, value: void 0 });
function stream(...collections) {
  if (collections.length === 1) {
    const collection = collections[0];
    if (collection instanceof StreamImpl) {
      return collection;
    }
    if (isIterable(collection)) {
      return new StreamImpl(() => collection[Symbol.iterator](), (iterator) => iterator.next());
    }
    if (typeof collection.length === "number") {
      return new StreamImpl(() => ({ index: 0 }), (state) => {
        if (state.index < collection.length) {
          return { done: false, value: collection[state.index++] };
        } else {
          return DONE_RESULT;
        }
      });
    }
  }
  if (collections.length > 1) {
    return new StreamImpl(() => ({ collIndex: 0, arrIndex: 0 }), (state) => {
      do {
        if (state.iterator) {
          const next = state.iterator.next();
          if (!next.done) {
            return next;
          }
          state.iterator = void 0;
        }
        if (state.array) {
          if (state.arrIndex < state.array.length) {
            return { done: false, value: state.array[state.arrIndex++] };
          }
          state.array = void 0;
          state.arrIndex = 0;
        }
        if (state.collIndex < collections.length) {
          const collection = collections[state.collIndex++];
          if (isIterable(collection)) {
            state.iterator = collection[Symbol.iterator]();
          } else if (collection && typeof collection.length === "number") {
            state.array = collection;
          }
        }
      } while (state.iterator || state.array || state.collIndex < collections.length);
      return DONE_RESULT;
    });
  }
  return EMPTY_STREAM;
}
var TreeStreamImpl = class extends StreamImpl {
  constructor(root2, children, options) {
    super(() => ({
      iterators: (options === null || options === void 0 ? void 0 : options.includeRoot) ? [[root2][Symbol.iterator]()] : [children(root2)[Symbol.iterator]()],
      pruned: false
    }), (state) => {
      if (state.pruned) {
        state.iterators.pop();
        state.pruned = false;
      }
      while (state.iterators.length > 0) {
        const iterator = state.iterators[state.iterators.length - 1];
        const next = iterator.next();
        if (next.done) {
          state.iterators.pop();
        } else {
          state.iterators.push(children(next.value)[Symbol.iterator]());
          return next;
        }
      }
      return DONE_RESULT;
    });
  }
  iterator() {
    const iterator = {
      state: this.startFn(),
      next: () => this.nextFn(iterator.state),
      prune: () => {
        iterator.state.pruned = true;
      },
      [Symbol.iterator]: () => iterator
    };
    return iterator;
  }
};
var Reduction;
(function(Reduction2) {
  function sum2(stream2) {
    return stream2.reduce((a, b) => a + b, 0);
  }
  Reduction2.sum = sum2;
  function product(stream2) {
    return stream2.reduce((a, b) => a * b, 0);
  }
  Reduction2.product = product;
  function min2(stream2) {
    return stream2.reduce((a, b) => Math.min(a, b));
  }
  Reduction2.min = min2;
  function max2(stream2) {
    return stream2.reduce((a, b) => Math.max(a, b));
  }
  Reduction2.max = max2;
})(Reduction || (Reduction = {}));

// node_modules/langium/lib/utils/cst-utils.js
function streamCst(node) {
  return new TreeStreamImpl(node, (element) => {
    if (isCompositeCstNode(element)) {
      return element.content;
    } else {
      return [];
    }
  }, { includeRoot: true });
}
function flattenCst(node) {
  return streamCst(node).filter(isLeafCstNode);
}
function isChildNode(child, parent2) {
  while (child.container) {
    child = child.container;
    if (child === parent2) {
      return true;
    }
  }
  return false;
}
function tokenToRange(token) {
  return {
    start: {
      character: token.startColumn - 1,
      line: token.startLine - 1
    },
    end: {
      character: token.endColumn,
      // endColumn uses the correct index
      line: token.endLine - 1
    }
  };
}
function toDocumentSegment(node) {
  if (!node) {
    return void 0;
  }
  const { offset, end, range: range2 } = node;
  return {
    range: range2,
    offset,
    end,
    length: end - offset
  };
}
var RangeComparison;
(function(RangeComparison2) {
  RangeComparison2[RangeComparison2["Before"] = 0] = "Before";
  RangeComparison2[RangeComparison2["After"] = 1] = "After";
  RangeComparison2[RangeComparison2["OverlapFront"] = 2] = "OverlapFront";
  RangeComparison2[RangeComparison2["OverlapBack"] = 3] = "OverlapBack";
  RangeComparison2[RangeComparison2["Inside"] = 4] = "Inside";
  RangeComparison2[RangeComparison2["Outside"] = 5] = "Outside";
})(RangeComparison || (RangeComparison = {}));
function compareRange(range2, to) {
  if (range2.end.line < to.start.line || range2.end.line === to.start.line && range2.end.character <= to.start.character) {
    return RangeComparison.Before;
  } else if (range2.start.line > to.end.line || range2.start.line === to.end.line && range2.start.character >= to.end.character) {
    return RangeComparison.After;
  }
  const startInside = range2.start.line > to.start.line || range2.start.line === to.start.line && range2.start.character >= to.start.character;
  const endInside = range2.end.line < to.end.line || range2.end.line === to.end.line && range2.end.character <= to.end.character;
  if (startInside && endInside) {
    return RangeComparison.Inside;
  } else if (startInside) {
    return RangeComparison.OverlapBack;
  } else if (endInside) {
    return RangeComparison.OverlapFront;
  } else {
    return RangeComparison.Outside;
  }
}
function inRange(range2, to) {
  const comparison = compareRange(range2, to);
  return comparison > RangeComparison.After;
}
var DefaultNameRegexp = /^[\w\p{L}]$/u;
function findDeclarationNodeAtOffset(cstNode, offset, nameRegexp = DefaultNameRegexp) {
  if (cstNode) {
    if (offset > 0) {
      const localOffset = offset - cstNode.offset;
      const textAtOffset = cstNode.text.charAt(localOffset);
      if (!nameRegexp.test(textAtOffset)) {
        offset--;
      }
    }
    return findLeafNodeAtOffset(cstNode, offset);
  }
  return void 0;
}
function findCommentNode(cstNode, commentNames) {
  if (cstNode) {
    const previous = getPreviousNode(cstNode, true);
    if (previous && isCommentNode(previous, commentNames)) {
      return previous;
    }
    if (isRootCstNode(cstNode)) {
      const endIndex = cstNode.content.findIndex((e) => !e.hidden);
      for (let i = endIndex - 1; i >= 0; i--) {
        const child = cstNode.content[i];
        if (isCommentNode(child, commentNames)) {
          return child;
        }
      }
    }
  }
  return void 0;
}
function isCommentNode(cstNode, commentNames) {
  return isLeafCstNode(cstNode) && commentNames.includes(cstNode.tokenType.name);
}
function findLeafNodeAtOffset(node, offset) {
  if (isLeafCstNode(node)) {
    return node;
  } else if (isCompositeCstNode(node)) {
    const searchResult = binarySearch(node, offset, false);
    if (searchResult) {
      return findLeafNodeAtOffset(searchResult, offset);
    }
  }
  return void 0;
}
function findLeafNodeBeforeOffset(node, offset) {
  if (isLeafCstNode(node)) {
    return node;
  } else if (isCompositeCstNode(node)) {
    const searchResult = binarySearch(node, offset, true);
    if (searchResult) {
      return findLeafNodeBeforeOffset(searchResult, offset);
    }
  }
  return void 0;
}
function binarySearch(node, offset, closest) {
  let left = 0;
  let right = node.content.length - 1;
  let closestNode = void 0;
  while (left <= right) {
    const middle = Math.floor((left + right) / 2);
    const middleNode = node.content[middle];
    if (middleNode.offset <= offset && middleNode.end > offset) {
      return middleNode;
    }
    if (middleNode.end <= offset) {
      closestNode = closest ? middleNode : void 0;
      left = middle + 1;
    } else {
      right = middle - 1;
    }
  }
  return closestNode;
}
function getPreviousNode(node, hidden = true) {
  while (node.container) {
    const parent2 = node.container;
    let index = parent2.content.indexOf(node);
    while (index > 0) {
      index--;
      const previous = parent2.content[index];
      if (hidden || !previous.hidden) {
        return previous;
      }
    }
    node = parent2;
  }
  return void 0;
}
function getNextNode(node, hidden = true) {
  while (node.container) {
    const parent2 = node.container;
    let index = parent2.content.indexOf(node);
    const last2 = parent2.content.length - 1;
    while (index < last2) {
      index++;
      const next = parent2.content[index];
      if (hidden || !next.hidden) {
        return next;
      }
    }
    node = parent2;
  }
  return void 0;
}
function getStartlineNode(node) {
  if (node.range.start.character === 0) {
    return node;
  }
  const line = node.range.start.line;
  let last2 = node;
  let index;
  while (node.container) {
    const parent2 = node.container;
    const selfIndex = index !== null && index !== void 0 ? index : parent2.content.indexOf(node);
    if (selfIndex === 0) {
      node = parent2;
      index = void 0;
    } else {
      index = selfIndex - 1;
      node = parent2.content[index];
    }
    if (node.range.start.line !== line) {
      break;
    }
    last2 = node;
  }
  return last2;
}
function getInteriorNodes(start, end) {
  const commonParent = getCommonParent(start, end);
  if (!commonParent) {
    return [];
  }
  return commonParent.parent.content.slice(commonParent.a + 1, commonParent.b);
}
function getCommonParent(a, b) {
  const aParents = getParentChain(a);
  const bParents = getParentChain(b);
  let current;
  for (let i = 0; i < aParents.length && i < bParents.length; i++) {
    const aParent = aParents[i];
    const bParent = bParents[i];
    if (aParent.parent === bParent.parent) {
      current = {
        parent: aParent.parent,
        a: aParent.index,
        b: bParent.index
      };
    } else {
      break;
    }
  }
  return current;
}
function getParentChain(node) {
  const chain2 = [];
  while (node.container) {
    const parent2 = node.container;
    const index = parent2.content.indexOf(node);
    chain2.push({
      parent: parent2,
      index
    });
    node = parent2;
  }
  return chain2.reverse();
}

// node_modules/langium/lib/utils/grammar-utils.js
var grammar_utils_exports = {};
__export(grammar_utils_exports, {
  findAssignment: () => findAssignment,
  findNameAssignment: () => findNameAssignment,
  findNodeForKeyword: () => findNodeForKeyword,
  findNodeForProperty: () => findNodeForProperty,
  findNodesForKeyword: () => findNodesForKeyword,
  findNodesForKeywordInternal: () => findNodesForKeywordInternal,
  findNodesForProperty: () => findNodesForProperty,
  getActionAtElement: () => getActionAtElement,
  getActionType: () => getActionType,
  getAllReachableRules: () => getAllReachableRules,
  getCrossReferenceTerminal: () => getCrossReferenceTerminal,
  getEntryRule: () => getEntryRule,
  getExplicitRuleType: () => getExplicitRuleType,
  getHiddenRules: () => getHiddenRules,
  getRuleType: () => getRuleType,
  getRuleTypeName: () => getRuleTypeName,
  getTypeName: () => getTypeName,
  isArrayCardinality: () => isArrayCardinality,
  isArrayOperator: () => isArrayOperator,
  isCommentTerminal: () => isCommentTerminal,
  isDataType: () => isDataType,
  isDataTypeRule: () => isDataTypeRule,
  isOptionalCardinality: () => isOptionalCardinality,
  terminalRegex: () => terminalRegex
});

// node_modules/langium/lib/utils/errors.js
var ErrorWithLocation = class extends Error {
  constructor(node, message) {
    super(node ? `${message} at ${node.range.start.line}:${node.range.start.character}` : message);
  }
};
function assertUnreachable(_) {
  throw new Error("Error! The input value was not handled.");
}

// node_modules/langium/lib/languages/generated/ast.js
var ast_exports = {};
__export(ast_exports, {
  AbstractElement: () => AbstractElement,
  AbstractRule: () => AbstractRule,
  AbstractType: () => AbstractType,
  Action: () => Action,
  Alternatives: () => Alternatives,
  ArrayLiteral: () => ArrayLiteral,
  ArrayType: () => ArrayType,
  Assignment: () => Assignment,
  BooleanLiteral: () => BooleanLiteral,
  CharacterRange: () => CharacterRange,
  Condition: () => Condition,
  Conjunction: () => Conjunction,
  CrossReference: () => CrossReference,
  Disjunction: () => Disjunction,
  EndOfFile: () => EndOfFile,
  Grammar: () => Grammar,
  GrammarImport: () => GrammarImport,
  Group: () => Group,
  InferredType: () => InferredType,
  Interface: () => Interface,
  Keyword: () => Keyword,
  LangiumGrammarAstReflection: () => LangiumGrammarAstReflection,
  LangiumGrammarTerminals: () => LangiumGrammarTerminals,
  NamedArgument: () => NamedArgument,
  NegatedToken: () => NegatedToken,
  Negation: () => Negation,
  NumberLiteral: () => NumberLiteral,
  Parameter: () => Parameter,
  ParameterReference: () => ParameterReference,
  ParserRule: () => ParserRule,
  ReferenceType: () => ReferenceType,
  RegexToken: () => RegexToken,
  ReturnType: () => ReturnType,
  RuleCall: () => RuleCall,
  SimpleType: () => SimpleType,
  StringLiteral: () => StringLiteral,
  TerminalAlternatives: () => TerminalAlternatives,
  TerminalGroup: () => TerminalGroup,
  TerminalRule: () => TerminalRule,
  TerminalRuleCall: () => TerminalRuleCall,
  Type: () => Type,
  TypeAttribute: () => TypeAttribute,
  TypeDefinition: () => TypeDefinition,
  UnionType: () => UnionType,
  UnorderedGroup: () => UnorderedGroup,
  UntilToken: () => UntilToken,
  ValueLiteral: () => ValueLiteral,
  Wildcard: () => Wildcard,
  isAbstractElement: () => isAbstractElement,
  isAbstractRule: () => isAbstractRule,
  isAbstractType: () => isAbstractType,
  isAction: () => isAction,
  isAlternatives: () => isAlternatives,
  isArrayLiteral: () => isArrayLiteral,
  isArrayType: () => isArrayType,
  isAssignment: () => isAssignment,
  isBooleanLiteral: () => isBooleanLiteral,
  isCharacterRange: () => isCharacterRange,
  isCondition: () => isCondition,
  isConjunction: () => isConjunction,
  isCrossReference: () => isCrossReference,
  isDisjunction: () => isDisjunction,
  isEndOfFile: () => isEndOfFile,
  isFeatureName: () => isFeatureName,
  isGrammar: () => isGrammar,
  isGrammarImport: () => isGrammarImport,
  isGroup: () => isGroup,
  isInferredType: () => isInferredType,
  isInterface: () => isInterface,
  isKeyword: () => isKeyword,
  isNamedArgument: () => isNamedArgument,
  isNegatedToken: () => isNegatedToken,
  isNegation: () => isNegation,
  isNumberLiteral: () => isNumberLiteral,
  isParameter: () => isParameter,
  isParameterReference: () => isParameterReference,
  isParserRule: () => isParserRule,
  isPrimitiveType: () => isPrimitiveType,
  isReferenceType: () => isReferenceType,
  isRegexToken: () => isRegexToken,
  isReturnType: () => isReturnType,
  isRuleCall: () => isRuleCall,
  isSimpleType: () => isSimpleType,
  isStringLiteral: () => isStringLiteral,
  isTerminalAlternatives: () => isTerminalAlternatives,
  isTerminalGroup: () => isTerminalGroup,
  isTerminalRule: () => isTerminalRule,
  isTerminalRuleCall: () => isTerminalRuleCall,
  isType: () => isType,
  isTypeAttribute: () => isTypeAttribute,
  isTypeDefinition: () => isTypeDefinition,
  isUnionType: () => isUnionType,
  isUnorderedGroup: () => isUnorderedGroup,
  isUntilToken: () => isUntilToken,
  isValueLiteral: () => isValueLiteral,
  isWildcard: () => isWildcard,
  reflection: () => reflection
});
var LangiumGrammarTerminals = {
  ID: /\^?[_a-zA-Z][\w_]*/,
  STRING: /"(\\.|[^"\\])*"|'(\\.|[^'\\])*'/,
  NUMBER: /NaN|-?((\d*\.\d+|\d+)([Ee][+-]?\d+)?|Infinity)/,
  RegexLiteral: /\/(?![*+?])(?:[^\r\n\[/\\]|\\.|\[(?:[^\r\n\]\\]|\\.)*\])+\/[a-z]*/,
  WS: /\s+/,
  ML_COMMENT: /\/\*[\s\S]*?\*\//,
  SL_COMMENT: /\/\/[^\n\r]*/
};
var AbstractRule = "AbstractRule";
function isAbstractRule(item) {
  return reflection.isInstance(item, AbstractRule);
}
var AbstractType = "AbstractType";
function isAbstractType(item) {
  return reflection.isInstance(item, AbstractType);
}
var Condition = "Condition";
function isCondition(item) {
  return reflection.isInstance(item, Condition);
}
function isFeatureName(item) {
  return isPrimitiveType(item) || item === "current" || item === "entry" || item === "extends" || item === "false" || item === "fragment" || item === "grammar" || item === "hidden" || item === "import" || item === "interface" || item === "returns" || item === "terminal" || item === "true" || item === "type" || item === "infer" || item === "infers" || item === "with" || typeof item === "string" && /\^?[_a-zA-Z][\w_]*/.test(item);
}
function isPrimitiveType(item) {
  return item === "string" || item === "number" || item === "boolean" || item === "Date" || item === "bigint";
}
var TypeDefinition = "TypeDefinition";
function isTypeDefinition(item) {
  return reflection.isInstance(item, TypeDefinition);
}
var ValueLiteral = "ValueLiteral";
function isValueLiteral(item) {
  return reflection.isInstance(item, ValueLiteral);
}
var AbstractElement = "AbstractElement";
function isAbstractElement(item) {
  return reflection.isInstance(item, AbstractElement);
}
var ArrayLiteral = "ArrayLiteral";
function isArrayLiteral(item) {
  return reflection.isInstance(item, ArrayLiteral);
}
var ArrayType = "ArrayType";
function isArrayType(item) {
  return reflection.isInstance(item, ArrayType);
}
var BooleanLiteral = "BooleanLiteral";
function isBooleanLiteral(item) {
  return reflection.isInstance(item, BooleanLiteral);
}
var Conjunction = "Conjunction";
function isConjunction(item) {
  return reflection.isInstance(item, Conjunction);
}
var Disjunction = "Disjunction";
function isDisjunction(item) {
  return reflection.isInstance(item, Disjunction);
}
var Grammar = "Grammar";
function isGrammar(item) {
  return reflection.isInstance(item, Grammar);
}
var GrammarImport = "GrammarImport";
function isGrammarImport(item) {
  return reflection.isInstance(item, GrammarImport);
}
var InferredType = "InferredType";
function isInferredType(item) {
  return reflection.isInstance(item, InferredType);
}
var Interface = "Interface";
function isInterface(item) {
  return reflection.isInstance(item, Interface);
}
var NamedArgument = "NamedArgument";
function isNamedArgument(item) {
  return reflection.isInstance(item, NamedArgument);
}
var Negation = "Negation";
function isNegation(item) {
  return reflection.isInstance(item, Negation);
}
var NumberLiteral = "NumberLiteral";
function isNumberLiteral(item) {
  return reflection.isInstance(item, NumberLiteral);
}
var Parameter = "Parameter";
function isParameter(item) {
  return reflection.isInstance(item, Parameter);
}
var ParameterReference = "ParameterReference";
function isParameterReference(item) {
  return reflection.isInstance(item, ParameterReference);
}
var ParserRule = "ParserRule";
function isParserRule(item) {
  return reflection.isInstance(item, ParserRule);
}
var ReferenceType = "ReferenceType";
function isReferenceType(item) {
  return reflection.isInstance(item, ReferenceType);
}
var ReturnType = "ReturnType";
function isReturnType(item) {
  return reflection.isInstance(item, ReturnType);
}
var SimpleType = "SimpleType";
function isSimpleType(item) {
  return reflection.isInstance(item, SimpleType);
}
var StringLiteral = "StringLiteral";
function isStringLiteral(item) {
  return reflection.isInstance(item, StringLiteral);
}
var TerminalRule = "TerminalRule";
function isTerminalRule(item) {
  return reflection.isInstance(item, TerminalRule);
}
var Type = "Type";
function isType(item) {
  return reflection.isInstance(item, Type);
}
var TypeAttribute = "TypeAttribute";
function isTypeAttribute(item) {
  return reflection.isInstance(item, TypeAttribute);
}
var UnionType = "UnionType";
function isUnionType(item) {
  return reflection.isInstance(item, UnionType);
}
var Action = "Action";
function isAction(item) {
  return reflection.isInstance(item, Action);
}
var Alternatives = "Alternatives";
function isAlternatives(item) {
  return reflection.isInstance(item, Alternatives);
}
var Assignment = "Assignment";
function isAssignment(item) {
  return reflection.isInstance(item, Assignment);
}
var CharacterRange = "CharacterRange";
function isCharacterRange(item) {
  return reflection.isInstance(item, CharacterRange);
}
var CrossReference = "CrossReference";
function isCrossReference(item) {
  return reflection.isInstance(item, CrossReference);
}
var EndOfFile = "EndOfFile";
function isEndOfFile(item) {
  return reflection.isInstance(item, EndOfFile);
}
var Group = "Group";
function isGroup(item) {
  return reflection.isInstance(item, Group);
}
var Keyword = "Keyword";
function isKeyword(item) {
  return reflection.isInstance(item, Keyword);
}
var NegatedToken = "NegatedToken";
function isNegatedToken(item) {
  return reflection.isInstance(item, NegatedToken);
}
var RegexToken = "RegexToken";
function isRegexToken(item) {
  return reflection.isInstance(item, RegexToken);
}
var RuleCall = "RuleCall";
function isRuleCall(item) {
  return reflection.isInstance(item, RuleCall);
}
var TerminalAlternatives = "TerminalAlternatives";
function isTerminalAlternatives(item) {
  return reflection.isInstance(item, TerminalAlternatives);
}
var TerminalGroup = "TerminalGroup";
function isTerminalGroup(item) {
  return reflection.isInstance(item, TerminalGroup);
}
var TerminalRuleCall = "TerminalRuleCall";
function isTerminalRuleCall(item) {
  return reflection.isInstance(item, TerminalRuleCall);
}
var UnorderedGroup = "UnorderedGroup";
function isUnorderedGroup(item) {
  return reflection.isInstance(item, UnorderedGroup);
}
var UntilToken = "UntilToken";
function isUntilToken(item) {
  return reflection.isInstance(item, UntilToken);
}
var Wildcard = "Wildcard";
function isWildcard(item) {
  return reflection.isInstance(item, Wildcard);
}
var LangiumGrammarAstReflection = class extends AbstractAstReflection {
  getAllTypes() {
    return [AbstractElement, AbstractRule, AbstractType, Action, Alternatives, ArrayLiteral, ArrayType, Assignment, BooleanLiteral, CharacterRange, Condition, Conjunction, CrossReference, Disjunction, EndOfFile, Grammar, GrammarImport, Group, InferredType, Interface, Keyword, NamedArgument, NegatedToken, Negation, NumberLiteral, Parameter, ParameterReference, ParserRule, ReferenceType, RegexToken, ReturnType, RuleCall, SimpleType, StringLiteral, TerminalAlternatives, TerminalGroup, TerminalRule, TerminalRuleCall, Type, TypeAttribute, TypeDefinition, UnionType, UnorderedGroup, UntilToken, ValueLiteral, Wildcard];
  }
  computeIsSubtype(subtype, supertype) {
    switch (subtype) {
      case Action:
      case Alternatives:
      case Assignment:
      case CharacterRange:
      case CrossReference:
      case EndOfFile:
      case Group:
      case Keyword:
      case NegatedToken:
      case RegexToken:
      case RuleCall:
      case TerminalAlternatives:
      case TerminalGroup:
      case TerminalRuleCall:
      case UnorderedGroup:
      case UntilToken:
      case Wildcard: {
        return this.isSubtype(AbstractElement, supertype);
      }
      case ArrayLiteral:
      case NumberLiteral:
      case StringLiteral: {
        return this.isSubtype(ValueLiteral, supertype);
      }
      case ArrayType:
      case ReferenceType:
      case SimpleType:
      case UnionType: {
        return this.isSubtype(TypeDefinition, supertype);
      }
      case BooleanLiteral: {
        return this.isSubtype(Condition, supertype) || this.isSubtype(ValueLiteral, supertype);
      }
      case Conjunction:
      case Disjunction:
      case Negation:
      case ParameterReference: {
        return this.isSubtype(Condition, supertype);
      }
      case InferredType:
      case Interface:
      case Type: {
        return this.isSubtype(AbstractType, supertype);
      }
      case ParserRule: {
        return this.isSubtype(AbstractRule, supertype) || this.isSubtype(AbstractType, supertype);
      }
      case TerminalRule: {
        return this.isSubtype(AbstractRule, supertype);
      }
      default: {
        return false;
      }
    }
  }
  getReferenceType(refInfo) {
    const referenceId = `${refInfo.container.$type}:${refInfo.property}`;
    switch (referenceId) {
      case "Action:type":
      case "CrossReference:type":
      case "Interface:superTypes":
      case "ParserRule:returnType":
      case "SimpleType:typeRef": {
        return AbstractType;
      }
      case "Grammar:hiddenTokens":
      case "ParserRule:hiddenTokens":
      case "RuleCall:rule": {
        return AbstractRule;
      }
      case "Grammar:usedGrammars": {
        return Grammar;
      }
      case "NamedArgument:parameter":
      case "ParameterReference:parameter": {
        return Parameter;
      }
      case "TerminalRuleCall:rule": {
        return TerminalRule;
      }
      default: {
        throw new Error(`${referenceId} is not a valid reference id.`);
      }
    }
  }
  getTypeMetaData(type) {
    switch (type) {
      case AbstractElement: {
        return {
          name: AbstractElement,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" }
          ]
        };
      }
      case ArrayLiteral: {
        return {
          name: ArrayLiteral,
          properties: [
            { name: "elements", defaultValue: [] }
          ]
        };
      }
      case ArrayType: {
        return {
          name: ArrayType,
          properties: [
            { name: "elementType" }
          ]
        };
      }
      case BooleanLiteral: {
        return {
          name: BooleanLiteral,
          properties: [
            { name: "true", defaultValue: false }
          ]
        };
      }
      case Conjunction: {
        return {
          name: Conjunction,
          properties: [
            { name: "left" },
            { name: "right" }
          ]
        };
      }
      case Disjunction: {
        return {
          name: Disjunction,
          properties: [
            { name: "left" },
            { name: "right" }
          ]
        };
      }
      case Grammar: {
        return {
          name: Grammar,
          properties: [
            { name: "definesHiddenTokens", defaultValue: false },
            { name: "hiddenTokens", defaultValue: [] },
            { name: "imports", defaultValue: [] },
            { name: "interfaces", defaultValue: [] },
            { name: "isDeclared", defaultValue: false },
            { name: "name" },
            { name: "rules", defaultValue: [] },
            { name: "types", defaultValue: [] },
            { name: "usedGrammars", defaultValue: [] }
          ]
        };
      }
      case GrammarImport: {
        return {
          name: GrammarImport,
          properties: [
            { name: "path" }
          ]
        };
      }
      case InferredType: {
        return {
          name: InferredType,
          properties: [
            { name: "name" }
          ]
        };
      }
      case Interface: {
        return {
          name: Interface,
          properties: [
            { name: "attributes", defaultValue: [] },
            { name: "name" },
            { name: "superTypes", defaultValue: [] }
          ]
        };
      }
      case NamedArgument: {
        return {
          name: NamedArgument,
          properties: [
            { name: "calledByName", defaultValue: false },
            { name: "parameter" },
            { name: "value" }
          ]
        };
      }
      case Negation: {
        return {
          name: Negation,
          properties: [
            { name: "value" }
          ]
        };
      }
      case NumberLiteral: {
        return {
          name: NumberLiteral,
          properties: [
            { name: "value" }
          ]
        };
      }
      case Parameter: {
        return {
          name: Parameter,
          properties: [
            { name: "name" }
          ]
        };
      }
      case ParameterReference: {
        return {
          name: ParameterReference,
          properties: [
            { name: "parameter" }
          ]
        };
      }
      case ParserRule: {
        return {
          name: ParserRule,
          properties: [
            { name: "dataType" },
            { name: "definesHiddenTokens", defaultValue: false },
            { name: "definition" },
            { name: "entry", defaultValue: false },
            { name: "fragment", defaultValue: false },
            { name: "hiddenTokens", defaultValue: [] },
            { name: "inferredType" },
            { name: "name" },
            { name: "parameters", defaultValue: [] },
            { name: "returnType" },
            { name: "wildcard", defaultValue: false }
          ]
        };
      }
      case ReferenceType: {
        return {
          name: ReferenceType,
          properties: [
            { name: "referenceType" }
          ]
        };
      }
      case ReturnType: {
        return {
          name: ReturnType,
          properties: [
            { name: "name" }
          ]
        };
      }
      case SimpleType: {
        return {
          name: SimpleType,
          properties: [
            { name: "primitiveType" },
            { name: "stringType" },
            { name: "typeRef" }
          ]
        };
      }
      case StringLiteral: {
        return {
          name: StringLiteral,
          properties: [
            { name: "value" }
          ]
        };
      }
      case TerminalRule: {
        return {
          name: TerminalRule,
          properties: [
            { name: "definition" },
            { name: "fragment", defaultValue: false },
            { name: "hidden", defaultValue: false },
            { name: "name" },
            { name: "type" }
          ]
        };
      }
      case Type: {
        return {
          name: Type,
          properties: [
            { name: "name" },
            { name: "type" }
          ]
        };
      }
      case TypeAttribute: {
        return {
          name: TypeAttribute,
          properties: [
            { name: "defaultValue" },
            { name: "isOptional", defaultValue: false },
            { name: "name" },
            { name: "type" }
          ]
        };
      }
      case UnionType: {
        return {
          name: UnionType,
          properties: [
            { name: "types", defaultValue: [] }
          ]
        };
      }
      case Action: {
        return {
          name: Action,
          properties: [
            { name: "cardinality" },
            { name: "feature" },
            { name: "inferredType" },
            { name: "lookahead" },
            { name: "operator" },
            { name: "type" }
          ]
        };
      }
      case Alternatives: {
        return {
          name: Alternatives,
          properties: [
            { name: "cardinality" },
            { name: "elements", defaultValue: [] },
            { name: "lookahead" }
          ]
        };
      }
      case Assignment: {
        return {
          name: Assignment,
          properties: [
            { name: "cardinality" },
            { name: "feature" },
            { name: "lookahead" },
            { name: "operator" },
            { name: "terminal" }
          ]
        };
      }
      case CharacterRange: {
        return {
          name: CharacterRange,
          properties: [
            { name: "cardinality" },
            { name: "left" },
            { name: "lookahead" },
            { name: "right" }
          ]
        };
      }
      case CrossReference: {
        return {
          name: CrossReference,
          properties: [
            { name: "cardinality" },
            { name: "deprecatedSyntax", defaultValue: false },
            { name: "lookahead" },
            { name: "terminal" },
            { name: "type" }
          ]
        };
      }
      case EndOfFile: {
        return {
          name: EndOfFile,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" }
          ]
        };
      }
      case Group: {
        return {
          name: Group,
          properties: [
            { name: "cardinality" },
            { name: "elements", defaultValue: [] },
            { name: "guardCondition" },
            { name: "lookahead" }
          ]
        };
      }
      case Keyword: {
        return {
          name: Keyword,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" },
            { name: "value" }
          ]
        };
      }
      case NegatedToken: {
        return {
          name: NegatedToken,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" },
            { name: "terminal" }
          ]
        };
      }
      case RegexToken: {
        return {
          name: RegexToken,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" },
            { name: "regex" }
          ]
        };
      }
      case RuleCall: {
        return {
          name: RuleCall,
          properties: [
            { name: "arguments", defaultValue: [] },
            { name: "cardinality" },
            { name: "lookahead" },
            { name: "rule" }
          ]
        };
      }
      case TerminalAlternatives: {
        return {
          name: TerminalAlternatives,
          properties: [
            { name: "cardinality" },
            { name: "elements", defaultValue: [] },
            { name: "lookahead" }
          ]
        };
      }
      case TerminalGroup: {
        return {
          name: TerminalGroup,
          properties: [
            { name: "cardinality" },
            { name: "elements", defaultValue: [] },
            { name: "lookahead" }
          ]
        };
      }
      case TerminalRuleCall: {
        return {
          name: TerminalRuleCall,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" },
            { name: "rule" }
          ]
        };
      }
      case UnorderedGroup: {
        return {
          name: UnorderedGroup,
          properties: [
            { name: "cardinality" },
            { name: "elements", defaultValue: [] },
            { name: "lookahead" }
          ]
        };
      }
      case UntilToken: {
        return {
          name: UntilToken,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" },
            { name: "terminal" }
          ]
        };
      }
      case Wildcard: {
        return {
          name: Wildcard,
          properties: [
            { name: "cardinality" },
            { name: "lookahead" }
          ]
        };
      }
      default: {
        return {
          name: type,
          properties: []
        };
      }
    }
  }
};
var reflection = new LangiumGrammarAstReflection();

// node_modules/langium/lib/utils/ast-utils.js
var ast_utils_exports = {};
__export(ast_utils_exports, {
  assignMandatoryProperties: () => assignMandatoryProperties,
  copyAstNode: () => copyAstNode,
  findLocalReferences: () => findLocalReferences,
  findRootNode: () => findRootNode,
  getContainerOfType: () => getContainerOfType,
  getDocument: () => getDocument,
  hasContainerOfType: () => hasContainerOfType,
  linkContentToContainer: () => linkContentToContainer,
  streamAllContents: () => streamAllContents,
  streamAst: () => streamAst,
  streamContents: () => streamContents,
  streamReferences: () => streamReferences
});
function linkContentToContainer(node) {
  for (const [name, value] of Object.entries(node)) {
    if (!name.startsWith("$")) {
      if (Array.isArray(value)) {
        value.forEach((item, index) => {
          if (isAstNode(item)) {
            item.$container = node;
            item.$containerProperty = name;
            item.$containerIndex = index;
          }
        });
      } else if (isAstNode(value)) {
        value.$container = node;
        value.$containerProperty = name;
      }
    }
  }
}
function getContainerOfType(node, typePredicate) {
  let item = node;
  while (item) {
    if (typePredicate(item)) {
      return item;
    }
    item = item.$container;
  }
  return void 0;
}
function hasContainerOfType(node, predicate) {
  let item = node;
  while (item) {
    if (predicate(item)) {
      return true;
    }
    item = item.$container;
  }
  return false;
}
function getDocument(node) {
  const rootNode = findRootNode(node);
  const result2 = rootNode.$document;
  if (!result2) {
    throw new Error("AST node has no document.");
  }
  return result2;
}
function findRootNode(node) {
  while (node.$container) {
    node = node.$container;
  }
  return node;
}
function streamContents(node, options) {
  if (!node) {
    throw new Error("Node must be an AstNode.");
  }
  const range2 = options === null || options === void 0 ? void 0 : options.range;
  return new StreamImpl(() => ({
    keys: Object.keys(node),
    keyIndex: 0,
    arrayIndex: 0
  }), (state) => {
    while (state.keyIndex < state.keys.length) {
      const property2 = state.keys[state.keyIndex];
      if (!property2.startsWith("$")) {
        const value = node[property2];
        if (isAstNode(value)) {
          state.keyIndex++;
          if (isAstNodeInRange(value, range2)) {
            return { done: false, value };
          }
        } else if (Array.isArray(value)) {
          while (state.arrayIndex < value.length) {
            const index = state.arrayIndex++;
            const element = value[index];
            if (isAstNode(element) && isAstNodeInRange(element, range2)) {
              return { done: false, value: element };
            }
          }
          state.arrayIndex = 0;
        }
      }
      state.keyIndex++;
    }
    return DONE_RESULT;
  });
}
function streamAllContents(root2, options) {
  if (!root2) {
    throw new Error("Root node must be an AstNode.");
  }
  return new TreeStreamImpl(root2, (node) => streamContents(node, options));
}
function streamAst(root2, options) {
  if (!root2) {
    throw new Error("Root node must be an AstNode.");
  } else if ((options === null || options === void 0 ? void 0 : options.range) && !isAstNodeInRange(root2, options.range)) {
    return new TreeStreamImpl(root2, () => []);
  }
  return new TreeStreamImpl(root2, (node) => streamContents(node, options), { includeRoot: true });
}
function isAstNodeInRange(astNode, range2) {
  var _a;
  if (!range2) {
    return true;
  }
  const nodeRange = (_a = astNode.$cstNode) === null || _a === void 0 ? void 0 : _a.range;
  if (!nodeRange) {
    return false;
  }
  return inRange(nodeRange, range2);
}
function streamReferences(node) {
  return new StreamImpl(() => ({
    keys: Object.keys(node),
    keyIndex: 0,
    arrayIndex: 0
  }), (state) => {
    while (state.keyIndex < state.keys.length) {
      const property2 = state.keys[state.keyIndex];
      if (!property2.startsWith("$")) {
        const value = node[property2];
        if (isReference(value)) {
          state.keyIndex++;
          return { done: false, value: { reference: value, container: node, property: property2 } };
        } else if (Array.isArray(value)) {
          while (state.arrayIndex < value.length) {
            const index = state.arrayIndex++;
            const element = value[index];
            if (isReference(element)) {
              return { done: false, value: { reference: element, container: node, property: property2, index } };
            }
          }
          state.arrayIndex = 0;
        }
      }
      state.keyIndex++;
    }
    return DONE_RESULT;
  });
}
function findLocalReferences(targetNode, lookup = getDocument(targetNode).parseResult.value) {
  const refs = [];
  streamAst(lookup).forEach((node) => {
    streamReferences(node).forEach((refInfo) => {
      if (refInfo.reference.ref === targetNode) {
        refs.push(refInfo.reference);
      }
    });
  });
  return stream(refs);
}
function assignMandatoryProperties(reflection2, node) {
  const typeMetaData = reflection2.getTypeMetaData(node.$type);
  const genericNode = node;
  for (const property2 of typeMetaData.properties) {
    if (property2.defaultValue !== void 0 && genericNode[property2.name] === void 0) {
      genericNode[property2.name] = copyDefaultValue(property2.defaultValue);
    }
  }
}
function copyDefaultValue(propertyType) {
  if (Array.isArray(propertyType)) {
    return [...propertyType.map(copyDefaultValue)];
  } else {
    return propertyType;
  }
}
function copyAstNode(node, buildReference) {
  const copy = { $type: node.$type };
  for (const [name, value] of Object.entries(node)) {
    if (!name.startsWith("$")) {
      if (isAstNode(value)) {
        copy[name] = copyAstNode(value, buildReference);
      } else if (isReference(value)) {
        copy[name] = buildReference(copy, name, value.$refNode, value.$refText);
      } else if (Array.isArray(value)) {
        const copiedArray = [];
        for (const element of value) {
          if (isAstNode(element)) {
            copiedArray.push(copyAstNode(element, buildReference));
          } else if (isReference(element)) {
            copiedArray.push(buildReference(copy, name, element.$refNode, element.$refText));
          } else {
            copiedArray.push(element);
          }
        }
        copy[name] = copiedArray;
      } else {
        copy[name] = value;
      }
    }
  }
  linkContentToContainer(copy);
  return copy;
}

// node_modules/langium/lib/utils/regexp-utils.js
var regexp_utils_exports = {};
__export(regexp_utils_exports, {
  NEWLINE_REGEXP: () => NEWLINE_REGEXP,
  escapeRegExp: () => escapeRegExp,
  getCaseInsensitivePattern: () => getCaseInsensitivePattern,
  getTerminalParts: () => getTerminalParts,
  isMultilineComment: () => isMultilineComment,
  isWhitespace: () => isWhitespace,
  partialMatches: () => partialMatches,
  partialRegExp: () => partialRegExp,
  whitespaceCharacters: () => whitespaceCharacters
});

// node_modules/@chevrotain/regexp-to-ast/lib/src/utils.js
function cc(char) {
  return char.charCodeAt(0);
}
function insertToSet(item, set2) {
  if (Array.isArray(item)) {
    item.forEach(function(subItem) {
      set2.push(subItem);
    });
  } else {
    set2.push(item);
  }
}
function addFlag(flagObj, flagKey) {
  if (flagObj[flagKey] === true) {
    throw "duplicate flag " + flagKey;
  }
  const x = flagObj[flagKey];
  flagObj[flagKey] = true;
}
function ASSERT_EXISTS(obj) {
  if (obj === void 0) {
    throw Error("Internal Error - Should never get here!");
  }
  return true;
}
function ASSERT_NEVER_REACH_HERE() {
  throw Error("Internal Error - Should never get here!");
}
function isCharacter(obj) {
  return obj["type"] === "Character";
}

// node_modules/@chevrotain/regexp-to-ast/lib/src/character-classes.js
var digitsCharCodes = [];
for (let i = cc("0"); i <= cc("9"); i++) {
  digitsCharCodes.push(i);
}
var wordCharCodes = [cc("_")].concat(digitsCharCodes);
for (let i = cc("a"); i <= cc("z"); i++) {
  wordCharCodes.push(i);
}
for (let i = cc("A"); i <= cc("Z"); i++) {
  wordCharCodes.push(i);
}
var whitespaceCodes = [
  cc(" "),
  cc("\f"),
  cc("\n"),
  cc("\r"),
  cc("	"),
  cc("\v"),
  cc("	"),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc(" "),
  cc("\u2028"),
  cc("\u2029"),
  cc(" "),
  cc(" "),
  cc("　"),
  cc("\uFEFF")
];

// node_modules/@chevrotain/regexp-to-ast/lib/src/regexp-parser.js
var hexDigitPattern = /[0-9a-fA-F]/;
var decimalPattern = /[0-9]/;
var decimalPatternNoZero = /[1-9]/;
var RegExpParser = class {
  constructor() {
    this.idx = 0;
    this.input = "";
    this.groupIdx = 0;
  }
  saveState() {
    return {
      idx: this.idx,
      input: this.input,
      groupIdx: this.groupIdx
    };
  }
  restoreState(newState2) {
    this.idx = newState2.idx;
    this.input = newState2.input;
    this.groupIdx = newState2.groupIdx;
  }
  pattern(input) {
    this.idx = 0;
    this.input = input;
    this.groupIdx = 0;
    this.consumeChar("/");
    const value = this.disjunction();
    this.consumeChar("/");
    const flags = {
      type: "Flags",
      loc: { begin: this.idx, end: input.length },
      global: false,
      ignoreCase: false,
      multiLine: false,
      unicode: false,
      sticky: false
    };
    while (this.isRegExpFlag()) {
      switch (this.popChar()) {
        case "g":
          addFlag(flags, "global");
          break;
        case "i":
          addFlag(flags, "ignoreCase");
          break;
        case "m":
          addFlag(flags, "multiLine");
          break;
        case "u":
          addFlag(flags, "unicode");
          break;
        case "y":
          addFlag(flags, "sticky");
          break;
      }
    }
    if (this.idx !== this.input.length) {
      throw Error("Redundant input: " + this.input.substring(this.idx));
    }
    return {
      type: "Pattern",
      flags,
      value,
      loc: this.loc(0)
    };
  }
  disjunction() {
    const alts = [];
    const begin = this.idx;
    alts.push(this.alternative());
    while (this.peekChar() === "|") {
      this.consumeChar("|");
      alts.push(this.alternative());
    }
    return { type: "Disjunction", value: alts, loc: this.loc(begin) };
  }
  alternative() {
    const terms = [];
    const begin = this.idx;
    while (this.isTerm()) {
      terms.push(this.term());
    }
    return { type: "Alternative", value: terms, loc: this.loc(begin) };
  }
  term() {
    if (this.isAssertion()) {
      return this.assertion();
    } else {
      return this.atom();
    }
  }
  assertion() {
    const begin = this.idx;
    switch (this.popChar()) {
      case "^":
        return {
          type: "StartAnchor",
          loc: this.loc(begin)
        };
      case "$":
        return { type: "EndAnchor", loc: this.loc(begin) };
      case "\\":
        switch (this.popChar()) {
          case "b":
            return {
              type: "WordBoundary",
              loc: this.loc(begin)
            };
          case "B":
            return {
              type: "NonWordBoundary",
              loc: this.loc(begin)
            };
        }
        throw Error("Invalid Assertion Escape");
      case "(":
        this.consumeChar("?");
        let type;
        switch (this.popChar()) {
          case "=":
            type = "Lookahead";
            break;
          case "!":
            type = "NegativeLookahead";
            break;
        }
        ASSERT_EXISTS(type);
        const disjunction = this.disjunction();
        this.consumeChar(")");
        return {
          type,
          value: disjunction,
          loc: this.loc(begin)
        };
    }
    return ASSERT_NEVER_REACH_HERE();
  }
  quantifier(isBacktracking = false) {
    let range2 = void 0;
    const begin = this.idx;
    switch (this.popChar()) {
      case "*":
        range2 = {
          atLeast: 0,
          atMost: Infinity
        };
        break;
      case "+":
        range2 = {
          atLeast: 1,
          atMost: Infinity
        };
        break;
      case "?":
        range2 = {
          atLeast: 0,
          atMost: 1
        };
        break;
      case "{":
        const atLeast = this.integerIncludingZero();
        switch (this.popChar()) {
          case "}":
            range2 = {
              atLeast,
              atMost: atLeast
            };
            break;
          case ",":
            let atMost;
            if (this.isDigit()) {
              atMost = this.integerIncludingZero();
              range2 = {
                atLeast,
                atMost
              };
            } else {
              range2 = {
                atLeast,
                atMost: Infinity
              };
            }
            this.consumeChar("}");
            break;
        }
        if (isBacktracking === true && range2 === void 0) {
          return void 0;
        }
        ASSERT_EXISTS(range2);
        break;
    }
    if (isBacktracking === true && range2 === void 0) {
      return void 0;
    }
    if (ASSERT_EXISTS(range2)) {
      if (this.peekChar(0) === "?") {
        this.consumeChar("?");
        range2.greedy = false;
      } else {
        range2.greedy = true;
      }
      range2.type = "Quantifier";
      range2.loc = this.loc(begin);
      return range2;
    }
  }
  atom() {
    let atom2;
    const begin = this.idx;
    switch (this.peekChar()) {
      case ".":
        atom2 = this.dotAll();
        break;
      case "\\":
        atom2 = this.atomEscape();
        break;
      case "[":
        atom2 = this.characterClass();
        break;
      case "(":
        atom2 = this.group();
        break;
    }
    if (atom2 === void 0 && this.isPatternCharacter()) {
      atom2 = this.patternCharacter();
    }
    if (ASSERT_EXISTS(atom2)) {
      atom2.loc = this.loc(begin);
      if (this.isQuantifier()) {
        atom2.quantifier = this.quantifier();
      }
      return atom2;
    }
    return ASSERT_NEVER_REACH_HERE();
  }
  dotAll() {
    this.consumeChar(".");
    return {
      type: "Set",
      complement: true,
      value: [cc("\n"), cc("\r"), cc("\u2028"), cc("\u2029")]
    };
  }
  atomEscape() {
    this.consumeChar("\\");
    switch (this.peekChar()) {
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
        return this.decimalEscapeAtom();
      case "d":
      case "D":
      case "s":
      case "S":
      case "w":
      case "W":
        return this.characterClassEscape();
      case "f":
      case "n":
      case "r":
      case "t":
      case "v":
        return this.controlEscapeAtom();
      case "c":
        return this.controlLetterEscapeAtom();
      case "0":
        return this.nulCharacterAtom();
      case "x":
        return this.hexEscapeSequenceAtom();
      case "u":
        return this.regExpUnicodeEscapeSequenceAtom();
      default:
        return this.identityEscapeAtom();
    }
  }
  decimalEscapeAtom() {
    const value = this.positiveInteger();
    return { type: "GroupBackReference", value };
  }
  characterClassEscape() {
    let set2;
    let complement = false;
    switch (this.popChar()) {
      case "d":
        set2 = digitsCharCodes;
        break;
      case "D":
        set2 = digitsCharCodes;
        complement = true;
        break;
      case "s":
        set2 = whitespaceCodes;
        break;
      case "S":
        set2 = whitespaceCodes;
        complement = true;
        break;
      case "w":
        set2 = wordCharCodes;
        break;
      case "W":
        set2 = wordCharCodes;
        complement = true;
        break;
    }
    if (ASSERT_EXISTS(set2)) {
      return { type: "Set", value: set2, complement };
    }
    return ASSERT_NEVER_REACH_HERE();
  }
  controlEscapeAtom() {
    let escapeCode;
    switch (this.popChar()) {
      case "f":
        escapeCode = cc("\f");
        break;
      case "n":
        escapeCode = cc("\n");
        break;
      case "r":
        escapeCode = cc("\r");
        break;
      case "t":
        escapeCode = cc("	");
        break;
      case "v":
        escapeCode = cc("\v");
        break;
    }
    if (ASSERT_EXISTS(escapeCode)) {
      return { type: "Character", value: escapeCode };
    }
    return ASSERT_NEVER_REACH_HERE();
  }
  controlLetterEscapeAtom() {
    this.consumeChar("c");
    const letter = this.popChar();
    if (/[a-zA-Z]/.test(letter) === false) {
      throw Error("Invalid ");
    }
    const letterCode = letter.toUpperCase().charCodeAt(0) - 64;
    return { type: "Character", value: letterCode };
  }
  nulCharacterAtom() {
    this.consumeChar("0");
    return { type: "Character", value: cc("\0") };
  }
  hexEscapeSequenceAtom() {
    this.consumeChar("x");
    return this.parseHexDigits(2);
  }
  regExpUnicodeEscapeSequenceAtom() {
    this.consumeChar("u");
    return this.parseHexDigits(4);
  }
  identityEscapeAtom() {
    const escapedChar = this.popChar();
    return { type: "Character", value: cc(escapedChar) };
  }
  classPatternCharacterAtom() {
    switch (this.peekChar()) {
      case "\n":
      case "\r":
      case "\u2028":
      case "\u2029":
      case "\\":
      case "]":
        throw Error("TBD");
      default:
        const nextChar = this.popChar();
        return { type: "Character", value: cc(nextChar) };
    }
  }
  characterClass() {
    const set2 = [];
    let complement = false;
    this.consumeChar("[");
    if (this.peekChar(0) === "^") {
      this.consumeChar("^");
      complement = true;
    }
    while (this.isClassAtom()) {
      const from = this.classAtom();
      const isFromSingleChar = from.type === "Character";
      if (isCharacter(from) && this.isRangeDash()) {
        this.consumeChar("-");
        const to = this.classAtom();
        const isToSingleChar = to.type === "Character";
        if (isCharacter(to)) {
          if (to.value < from.value) {
            throw Error("Range out of order in character class");
          }
          set2.push({ from: from.value, to: to.value });
        } else {
          insertToSet(from.value, set2);
          set2.push(cc("-"));
          insertToSet(to.value, set2);
        }
      } else {
        insertToSet(from.value, set2);
      }
    }
    this.consumeChar("]");
    return { type: "Set", complement, value: set2 };
  }
  classAtom() {
    switch (this.peekChar()) {
      case "]":
      case "\n":
      case "\r":
      case "\u2028":
      case "\u2029":
        throw Error("TBD");
      case "\\":
        return this.classEscape();
      default:
        return this.classPatternCharacterAtom();
    }
  }
  classEscape() {
    this.consumeChar("\\");
    switch (this.peekChar()) {
      case "b":
        this.consumeChar("b");
        return { type: "Character", value: cc("\b") };
      case "d":
      case "D":
      case "s":
      case "S":
      case "w":
      case "W":
        return this.characterClassEscape();
      case "f":
      case "n":
      case "r":
      case "t":
      case "v":
        return this.controlEscapeAtom();
      case "c":
        return this.controlLetterEscapeAtom();
      case "0":
        return this.nulCharacterAtom();
      case "x":
        return this.hexEscapeSequenceAtom();
      case "u":
        return this.regExpUnicodeEscapeSequenceAtom();
      default:
        return this.identityEscapeAtom();
    }
  }
  group() {
    let capturing = true;
    this.consumeChar("(");
    switch (this.peekChar(0)) {
      case "?":
        this.consumeChar("?");
        this.consumeChar(":");
        capturing = false;
        break;
      default:
        this.groupIdx++;
        break;
    }
    const value = this.disjunction();
    this.consumeChar(")");
    const groupAst = {
      type: "Group",
      capturing,
      value
    };
    if (capturing) {
      groupAst["idx"] = this.groupIdx;
    }
    return groupAst;
  }
  positiveInteger() {
    let number = this.popChar();
    if (decimalPatternNoZero.test(number) === false) {
      throw Error("Expecting a positive integer");
    }
    while (decimalPattern.test(this.peekChar(0))) {
      number += this.popChar();
    }
    return parseInt(number, 10);
  }
  integerIncludingZero() {
    let number = this.popChar();
    if (decimalPattern.test(number) === false) {
      throw Error("Expecting an integer");
    }
    while (decimalPattern.test(this.peekChar(0))) {
      number += this.popChar();
    }
    return parseInt(number, 10);
  }
  patternCharacter() {
    const nextChar = this.popChar();
    switch (nextChar) {
      case "\n":
      case "\r":
      case "\u2028":
      case "\u2029":
      case "^":
      case "$":
      case "\\":
      case ".":
      case "*":
      case "+":
      case "?":
      case "(":
      case ")":
      case "[":
      case "|":
        throw Error("TBD");
      default:
        return { type: "Character", value: cc(nextChar) };
    }
  }
  isRegExpFlag() {
    switch (this.peekChar(0)) {
      case "g":
      case "i":
      case "m":
      case "u":
      case "y":
        return true;
      default:
        return false;
    }
  }
  isRangeDash() {
    return this.peekChar() === "-" && this.isClassAtom(1);
  }
  isDigit() {
    return decimalPattern.test(this.peekChar(0));
  }
  isClassAtom(howMuch = 0) {
    switch (this.peekChar(howMuch)) {
      case "]":
      case "\n":
      case "\r":
      case "\u2028":
      case "\u2029":
        return false;
      default:
        return true;
    }
  }
  isTerm() {
    return this.isAtom() || this.isAssertion();
  }
  isAtom() {
    if (this.isPatternCharacter()) {
      return true;
    }
    switch (this.peekChar(0)) {
      case ".":
      case "\\":
      case "[":
      case "(":
        return true;
      default:
        return false;
    }
  }
  isAssertion() {
    switch (this.peekChar(0)) {
      case "^":
      case "$":
        return true;
      case "\\":
        switch (this.peekChar(1)) {
          case "b":
          case "B":
            return true;
          default:
            return false;
        }
      case "(":
        return this.peekChar(1) === "?" && (this.peekChar(2) === "=" || this.peekChar(2) === "!");
      default:
        return false;
    }
  }
  isQuantifier() {
    const prevState = this.saveState();
    try {
      return this.quantifier(true) !== void 0;
    } catch (e) {
      return false;
    } finally {
      this.restoreState(prevState);
    }
  }
  isPatternCharacter() {
    switch (this.peekChar()) {
      case "^":
      case "$":
      case "\\":
      case ".":
      case "*":
      case "+":
      case "?":
      case "(":
      case ")":
      case "[":
      case "|":
      case "/":
      case "\n":
      case "\r":
      case "\u2028":
      case "\u2029":
        return false;
      default:
        return true;
    }
  }
  parseHexDigits(howMany) {
    let hexString = "";
    for (let i = 0; i < howMany; i++) {
      const hexChar = this.popChar();
      if (hexDigitPattern.test(hexChar) === false) {
        throw Error("Expecting a HexDecimal digits");
      }
      hexString += hexChar;
    }
    const charCode = parseInt(hexString, 16);
    return { type: "Character", value: charCode };
  }
  peekChar(howMuch = 0) {
    return this.input[this.idx + howMuch];
  }
  popChar() {
    const nextChar = this.peekChar(0);
    this.consumeChar(void 0);
    return nextChar;
  }
  consumeChar(char) {
    if (char !== void 0 && this.input[this.idx] !== char) {
      throw Error("Expected: '" + char + "' but found: '" + this.input[this.idx] + "' at offset: " + this.idx);
    }
    if (this.idx >= this.input.length) {
      throw Error("Unexpected end of input");
    }
    this.idx++;
  }
  loc(begin) {
    return { begin, end: this.idx };
  }
};

// node_modules/@chevrotain/regexp-to-ast/lib/src/base-regexp-visitor.js
var BaseRegExpVisitor = class {
  visitChildren(node) {
    for (const key in node) {
      const child = node[key];
      if (node.hasOwnProperty(key)) {
        if (child.type !== void 0) {
          this.visit(child);
        } else if (Array.isArray(child)) {
          child.forEach((subChild) => {
            this.visit(subChild);
          }, this);
        }
      }
    }
  }
  visit(node) {
    switch (node.type) {
      case "Pattern":
        this.visitPattern(node);
        break;
      case "Flags":
        this.visitFlags(node);
        break;
      case "Disjunction":
        this.visitDisjunction(node);
        break;
      case "Alternative":
        this.visitAlternative(node);
        break;
      case "StartAnchor":
        this.visitStartAnchor(node);
        break;
      case "EndAnchor":
        this.visitEndAnchor(node);
        break;
      case "WordBoundary":
        this.visitWordBoundary(node);
        break;
      case "NonWordBoundary":
        this.visitNonWordBoundary(node);
        break;
      case "Lookahead":
        this.visitLookahead(node);
        break;
      case "NegativeLookahead":
        this.visitNegativeLookahead(node);
        break;
      case "Character":
        this.visitCharacter(node);
        break;
      case "Set":
        this.visitSet(node);
        break;
      case "Group":
        this.visitGroup(node);
        break;
      case "GroupBackReference":
        this.visitGroupBackReference(node);
        break;
      case "Quantifier":
        this.visitQuantifier(node);
        break;
    }
    this.visitChildren(node);
  }
  visitPattern(node) {
  }
  visitFlags(node) {
  }
  visitDisjunction(node) {
  }
  visitAlternative(node) {
  }
  // Assertion
  visitStartAnchor(node) {
  }
  visitEndAnchor(node) {
  }
  visitWordBoundary(node) {
  }
  visitNonWordBoundary(node) {
  }
  visitLookahead(node) {
  }
  visitNegativeLookahead(node) {
  }
  // atoms
  visitCharacter(node) {
  }
  visitSet(node) {
  }
  visitGroup(node) {
  }
  visitGroupBackReference(node) {
  }
  visitQuantifier(node) {
  }
};

// node_modules/langium/lib/utils/regexp-utils.js
var NEWLINE_REGEXP = /\r?\n/gm;
var regexpParser = new RegExpParser();
var TerminalRegExpVisitor = class extends BaseRegExpVisitor {
  constructor() {
    super(...arguments);
    this.isStarting = true;
    this.endRegexpStack = [];
    this.multiline = false;
  }
  get endRegex() {
    return this.endRegexpStack.join("");
  }
  reset(regex) {
    this.multiline = false;
    this.regex = regex;
    this.startRegexp = "";
    this.isStarting = true;
    this.endRegexpStack = [];
  }
  visitGroup(node) {
    if (node.quantifier) {
      this.isStarting = false;
      this.endRegexpStack = [];
    }
  }
  visitCharacter(node) {
    const char = String.fromCharCode(node.value);
    if (!this.multiline && char === "\n") {
      this.multiline = true;
    }
    if (node.quantifier) {
      this.isStarting = false;
      this.endRegexpStack = [];
    } else {
      const escapedChar = escapeRegExp(char);
      this.endRegexpStack.push(escapedChar);
      if (this.isStarting) {
        this.startRegexp += escapedChar;
      }
    }
  }
  visitSet(node) {
    if (!this.multiline) {
      const set2 = this.regex.substring(node.loc.begin, node.loc.end);
      const regex = new RegExp(set2);
      this.multiline = Boolean("\n".match(regex));
    }
    if (node.quantifier) {
      this.isStarting = false;
      this.endRegexpStack = [];
    } else {
      const set2 = this.regex.substring(node.loc.begin, node.loc.end);
      this.endRegexpStack.push(set2);
      if (this.isStarting) {
        this.startRegexp += set2;
      }
    }
  }
  visitChildren(node) {
    if (node.type === "Group") {
      const group = node;
      if (group.quantifier) {
        return;
      }
    }
    super.visitChildren(node);
  }
};
var visitor = new TerminalRegExpVisitor();
function getTerminalParts(regexp) {
  try {
    if (typeof regexp !== "string") {
      regexp = regexp.source;
    }
    regexp = `/${regexp}/`;
    const pattern = regexpParser.pattern(regexp);
    const parts = [];
    for (const alternative of pattern.value.value) {
      visitor.reset(regexp);
      visitor.visit(alternative);
      parts.push({
        start: visitor.startRegexp,
        end: visitor.endRegex
      });
    }
    return parts;
  } catch (_a) {
    return [];
  }
}
function isMultilineComment(regexp) {
  try {
    if (typeof regexp === "string") {
      regexp = new RegExp(regexp);
    }
    regexp = regexp.toString();
    visitor.reset(regexp);
    visitor.visit(regexpParser.pattern(regexp));
    return visitor.multiline;
  } catch (_a) {
    return false;
  }
}
var whitespaceCharacters = "\f\n\r	\v              \u2028\u2029  　\uFEFF".split("");
function isWhitespace(value) {
  const regexp = typeof value === "string" ? new RegExp(value) : value;
  return whitespaceCharacters.some((ws) => regexp.test(ws));
}
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function getCaseInsensitivePattern(keyword) {
  return Array.prototype.map.call(keyword, (letter) => /\w/.test(letter) ? `[${letter.toLowerCase()}${letter.toUpperCase()}]` : escapeRegExp(letter)).join("");
}
function partialMatches(regex, input) {
  const partial2 = partialRegExp(regex);
  const match = input.match(partial2);
  return !!match && match[0].length > 0;
}
function partialRegExp(regex) {
  if (typeof regex === "string") {
    regex = new RegExp(regex);
  }
  const re = regex, source = regex.source;
  let i = 0;
  function process2() {
    let result2 = "", tmp;
    function appendRaw(nbChars) {
      result2 += source.substr(i, nbChars);
      i += nbChars;
    }
    function appendOptional(nbChars) {
      result2 += "(?:" + source.substr(i, nbChars) + "|$)";
      i += nbChars;
    }
    while (i < source.length) {
      switch (source[i]) {
        case "\\":
          switch (source[i + 1]) {
            case "c":
              appendOptional(3);
              break;
            case "x":
              appendOptional(4);
              break;
            case "u":
              if (re.unicode) {
                if (source[i + 2] === "{") {
                  appendOptional(source.indexOf("}", i) - i + 1);
                } else {
                  appendOptional(6);
                }
              } else {
                appendOptional(2);
              }
              break;
            case "p":
            case "P":
              if (re.unicode) {
                appendOptional(source.indexOf("}", i) - i + 1);
              } else {
                appendOptional(2);
              }
              break;
            case "k":
              appendOptional(source.indexOf(">", i) - i + 1);
              break;
            default:
              appendOptional(2);
              break;
          }
          break;
        case "[":
          tmp = /\[(?:\\.|.)*?\]/g;
          tmp.lastIndex = i;
          tmp = tmp.exec(source) || [];
          appendOptional(tmp[0].length);
          break;
        case "|":
        case "^":
        case "$":
        case "*":
        case "+":
        case "?":
          appendRaw(1);
          break;
        case "{":
          tmp = /\{\d+,?\d*\}/g;
          tmp.lastIndex = i;
          tmp = tmp.exec(source);
          if (tmp) {
            appendRaw(tmp[0].length);
          } else {
            appendOptional(1);
          }
          break;
        case "(":
          if (source[i + 1] === "?") {
            switch (source[i + 2]) {
              case ":":
                result2 += "(?:";
                i += 3;
                result2 += process2() + "|$)";
                break;
              case "=":
                result2 += "(?=";
                i += 3;
                result2 += process2() + ")";
                break;
              case "!":
                tmp = i;
                i += 3;
                process2();
                result2 += source.substr(tmp, i - tmp);
                break;
              case "<":
                switch (source[i + 3]) {
                  case "=":
                  case "!":
                    tmp = i;
                    i += 4;
                    process2();
                    result2 += source.substr(tmp, i - tmp);
                    break;
                  default:
                    appendRaw(source.indexOf(">", i) - i + 1);
                    result2 += process2() + "|$)";
                    break;
                }
                break;
            }
          } else {
            appendRaw(1);
            result2 += process2() + "|$)";
          }
          break;
        case ")":
          ++i;
          return result2;
        default:
          appendOptional(1);
          break;
      }
    }
    return result2;
  }
  return new RegExp(process2(), regex.flags);
}

// node_modules/langium/lib/utils/grammar-utils.js
function getEntryRule(grammar) {
  return grammar.rules.find((e) => isParserRule(e) && e.entry);
}
function getHiddenRules(grammar) {
  return grammar.rules.filter((e) => isTerminalRule(e) && e.hidden);
}
function getAllReachableRules(grammar, allTerminals) {
  const ruleNames = /* @__PURE__ */ new Set();
  const entryRule = getEntryRule(grammar);
  if (!entryRule) {
    return new Set(grammar.rules);
  }
  const topMostRules = [entryRule].concat(getHiddenRules(grammar));
  for (const rule of topMostRules) {
    ruleDfs(rule, ruleNames, allTerminals);
  }
  const rules = /* @__PURE__ */ new Set();
  for (const rule of grammar.rules) {
    if (ruleNames.has(rule.name) || isTerminalRule(rule) && rule.hidden) {
      rules.add(rule);
    }
  }
  return rules;
}
function ruleDfs(rule, visitedSet, allTerminals) {
  visitedSet.add(rule.name);
  streamAllContents(rule).forEach((node) => {
    if (isRuleCall(node) || allTerminals && isTerminalRuleCall(node)) {
      const refRule = node.rule.ref;
      if (refRule && !visitedSet.has(refRule.name)) {
        ruleDfs(refRule, visitedSet, allTerminals);
      }
    }
  });
}
function getCrossReferenceTerminal(crossRef) {
  if (crossRef.terminal) {
    return crossRef.terminal;
  } else if (crossRef.type.ref) {
    const nameAssigment = findNameAssignment(crossRef.type.ref);
    return nameAssigment === null || nameAssigment === void 0 ? void 0 : nameAssigment.terminal;
  }
  return void 0;
}
function isCommentTerminal(terminalRule) {
  return terminalRule.hidden && !isWhitespace(terminalRegex(terminalRule));
}
function findNodesForProperty(node, property2) {
  if (!node || !property2) {
    return [];
  }
  return findNodesForPropertyInternal(node, property2, node.astNode, true);
}
function findNodeForProperty(node, property2, index) {
  if (!node || !property2) {
    return void 0;
  }
  const nodes = findNodesForPropertyInternal(node, property2, node.astNode, true);
  if (nodes.length === 0) {
    return void 0;
  }
  if (index !== void 0) {
    index = Math.max(0, Math.min(index, nodes.length - 1));
  } else {
    index = 0;
  }
  return nodes[index];
}
function findNodesForPropertyInternal(node, property2, element, first2) {
  if (!first2) {
    const nodeFeature = getContainerOfType(node.grammarSource, isAssignment);
    if (nodeFeature && nodeFeature.feature === property2) {
      return [node];
    }
  }
  if (isCompositeCstNode(node) && node.astNode === element) {
    return node.content.flatMap((e) => findNodesForPropertyInternal(e, property2, element, false));
  }
  return [];
}
function findNodesForKeyword(node, keyword) {
  if (!node) {
    return [];
  }
  return findNodesForKeywordInternal(node, keyword, node === null || node === void 0 ? void 0 : node.astNode);
}
function findNodeForKeyword(node, keyword, index) {
  if (!node) {
    return void 0;
  }
  const nodes = findNodesForKeywordInternal(node, keyword, node === null || node === void 0 ? void 0 : node.astNode);
  if (nodes.length === 0) {
    return void 0;
  }
  if (index !== void 0) {
    index = Math.max(0, Math.min(index, nodes.length - 1));
  } else {
    index = 0;
  }
  return nodes[index];
}
function findNodesForKeywordInternal(node, keyword, element) {
  if (node.astNode !== element) {
    return [];
  }
  if (isKeyword(node.grammarSource) && node.grammarSource.value === keyword) {
    return [node];
  }
  const treeIterator = streamCst(node).iterator();
  let result2;
  const keywordNodes = [];
  do {
    result2 = treeIterator.next();
    if (!result2.done) {
      const childNode = result2.value;
      if (childNode.astNode === element) {
        if (isKeyword(childNode.grammarSource) && childNode.grammarSource.value === keyword) {
          keywordNodes.push(childNode);
        }
      } else {
        treeIterator.prune();
      }
    }
  } while (!result2.done);
  return keywordNodes;
}
function findAssignment(cstNode) {
  var _a;
  const astNode = cstNode.astNode;
  while (astNode === ((_a = cstNode.container) === null || _a === void 0 ? void 0 : _a.astNode)) {
    const assignment = getContainerOfType(cstNode.grammarSource, isAssignment);
    if (assignment) {
      return assignment;
    }
    cstNode = cstNode.container;
  }
  return void 0;
}
function findNameAssignment(type) {
  let startNode = type;
  if (isInferredType(startNode)) {
    if (isAction(startNode.$container)) {
      startNode = startNode.$container.$container;
    } else if (isParserRule(startNode.$container)) {
      startNode = startNode.$container;
    } else {
      assertUnreachable(startNode.$container);
    }
  }
  return findNameAssignmentInternal(type, startNode, /* @__PURE__ */ new Map());
}
function findNameAssignmentInternal(type, startNode, cache) {
  var _a;
  function go(node, refType) {
    let childAssignment = void 0;
    const parentAssignment = getContainerOfType(node, isAssignment);
    if (!parentAssignment) {
      childAssignment = findNameAssignmentInternal(refType, refType, cache);
    }
    cache.set(type, childAssignment);
    return childAssignment;
  }
  if (cache.has(type)) {
    return cache.get(type);
  }
  cache.set(type, void 0);
  for (const node of streamAllContents(startNode)) {
    if (isAssignment(node) && node.feature.toLowerCase() === "name") {
      cache.set(type, node);
      return node;
    } else if (isRuleCall(node) && isParserRule(node.rule.ref)) {
      return go(node, node.rule.ref);
    } else if (isSimpleType(node) && ((_a = node.typeRef) === null || _a === void 0 ? void 0 : _a.ref)) {
      return go(node, node.typeRef.ref);
    }
  }
  return void 0;
}
function getActionAtElement(element) {
  const parent2 = element.$container;
  if (isGroup(parent2)) {
    const elements = parent2.elements;
    const index = elements.indexOf(element);
    for (let i = index - 1; i >= 0; i--) {
      const item = elements[i];
      if (isAction(item)) {
        return item;
      } else {
        const action = streamAllContents(elements[i]).find(isAction);
        if (action) {
          return action;
        }
      }
    }
  }
  if (isAbstractElement(parent2)) {
    return getActionAtElement(parent2);
  } else {
    return void 0;
  }
}
function isOptionalCardinality(cardinality, element) {
  return cardinality === "?" || cardinality === "*" || isGroup(element) && Boolean(element.guardCondition);
}
function isArrayCardinality(cardinality) {
  return cardinality === "*" || cardinality === "+";
}
function isArrayOperator(operator) {
  return operator === "+=";
}
function isDataTypeRule(rule) {
  return isDataTypeRuleInternal(rule, /* @__PURE__ */ new Set());
}
function isDataTypeRuleInternal(rule, visited) {
  if (visited.has(rule)) {
    return true;
  } else {
    visited.add(rule);
  }
  for (const node of streamAllContents(rule)) {
    if (isRuleCall(node)) {
      if (!node.rule.ref) {
        return false;
      }
      if (isParserRule(node.rule.ref) && !isDataTypeRuleInternal(node.rule.ref, visited)) {
        return false;
      }
    } else if (isAssignment(node)) {
      return false;
    } else if (isAction(node)) {
      return false;
    }
  }
  return Boolean(rule.definition);
}
function isDataType(type) {
  return isDataTypeInternal(type.type, /* @__PURE__ */ new Set());
}
function isDataTypeInternal(type, visited) {
  if (visited.has(type)) {
    return true;
  } else {
    visited.add(type);
  }
  if (isArrayType(type)) {
    return false;
  } else if (isReferenceType(type)) {
    return false;
  } else if (isUnionType(type)) {
    return type.types.every((e) => isDataTypeInternal(e, visited));
  } else if (isSimpleType(type)) {
    if (type.primitiveType !== void 0) {
      return true;
    } else if (type.stringType !== void 0) {
      return true;
    } else if (type.typeRef !== void 0) {
      const ref = type.typeRef.ref;
      if (isType(ref)) {
        return isDataTypeInternal(ref.type, visited);
      } else {
        return false;
      }
    } else {
      return false;
    }
  } else {
    return false;
  }
}
function getExplicitRuleType(rule) {
  if (rule.inferredType) {
    return rule.inferredType.name;
  } else if (rule.dataType) {
    return rule.dataType;
  } else if (rule.returnType) {
    const refType = rule.returnType.ref;
    if (refType) {
      if (isParserRule(refType)) {
        return refType.name;
      } else if (isInterface(refType) || isType(refType)) {
        return refType.name;
      }
    }
  }
  return void 0;
}
function getTypeName(type) {
  var _a;
  if (isParserRule(type)) {
    return isDataTypeRule(type) ? type.name : (_a = getExplicitRuleType(type)) !== null && _a !== void 0 ? _a : type.name;
  } else if (isInterface(type) || isType(type) || isReturnType(type)) {
    return type.name;
  } else if (isAction(type)) {
    const actionType = getActionType(type);
    if (actionType) {
      return actionType;
    }
  } else if (isInferredType(type)) {
    return type.name;
  }
  throw new Error("Cannot get name of Unknown Type");
}
function getActionType(action) {
  var _a;
  if (action.inferredType) {
    return action.inferredType.name;
  } else if ((_a = action.type) === null || _a === void 0 ? void 0 : _a.ref) {
    return getTypeName(action.type.ref);
  }
  return void 0;
}
function getRuleTypeName(rule) {
  var _a, _b, _c;
  if (isTerminalRule(rule)) {
    return (_b = (_a = rule.type) === null || _a === void 0 ? void 0 : _a.name) !== null && _b !== void 0 ? _b : "string";
  } else {
    return isDataTypeRule(rule) ? rule.name : (_c = getExplicitRuleType(rule)) !== null && _c !== void 0 ? _c : rule.name;
  }
}
function getRuleType(rule) {
  var _a, _b, _c;
  if (isTerminalRule(rule)) {
    return (_b = (_a = rule.type) === null || _a === void 0 ? void 0 : _a.name) !== null && _b !== void 0 ? _b : "string";
  } else {
    return (_c = getExplicitRuleType(rule)) !== null && _c !== void 0 ? _c : rule.name;
  }
}
function terminalRegex(terminalRule) {
  const flags = {
    s: false,
    i: false,
    u: false
  };
  const source = abstractElementToRegex(terminalRule.definition, flags);
  const flagText = Object.entries(flags).filter(([, value]) => value).map(([name]) => name).join("");
  return new RegExp(source, flagText);
}
var WILDCARD = /[\s\S]/.source;
function abstractElementToRegex(element, flags) {
  if (isTerminalAlternatives(element)) {
    return terminalAlternativesToRegex(element);
  } else if (isTerminalGroup(element)) {
    return terminalGroupToRegex(element);
  } else if (isCharacterRange(element)) {
    return characterRangeToRegex(element);
  } else if (isTerminalRuleCall(element)) {
    const rule = element.rule.ref;
    if (!rule) {
      throw new Error("Missing rule reference.");
    }
    return withCardinality(abstractElementToRegex(rule.definition), {
      cardinality: element.cardinality,
      lookahead: element.lookahead
    });
  } else if (isNegatedToken(element)) {
    return negateTokenToRegex(element);
  } else if (isUntilToken(element)) {
    return untilTokenToRegex(element);
  } else if (isRegexToken(element)) {
    const lastSlash = element.regex.lastIndexOf("/");
    const source = element.regex.substring(1, lastSlash);
    const regexFlags = element.regex.substring(lastSlash + 1);
    if (flags) {
      flags.i = regexFlags.includes("i");
      flags.s = regexFlags.includes("s");
      flags.u = regexFlags.includes("u");
    }
    return withCardinality(source, {
      cardinality: element.cardinality,
      lookahead: element.lookahead,
      wrap: false
    });
  } else if (isWildcard(element)) {
    return withCardinality(WILDCARD, {
      cardinality: element.cardinality,
      lookahead: element.lookahead
    });
  } else {
    throw new Error(`Invalid terminal element: ${element === null || element === void 0 ? void 0 : element.$type}`);
  }
}
function terminalAlternativesToRegex(alternatives) {
  return withCardinality(alternatives.elements.map((e) => abstractElementToRegex(e)).join("|"), {
    cardinality: alternatives.cardinality,
    lookahead: alternatives.lookahead
  });
}
function terminalGroupToRegex(group) {
  return withCardinality(group.elements.map((e) => abstractElementToRegex(e)).join(""), {
    cardinality: group.cardinality,
    lookahead: group.lookahead
  });
}
function untilTokenToRegex(until) {
  return withCardinality(`${WILDCARD}*?${abstractElementToRegex(until.terminal)}`, {
    cardinality: until.cardinality,
    lookahead: until.lookahead
  });
}
function negateTokenToRegex(negate2) {
  return withCardinality(`(?!${abstractElementToRegex(negate2.terminal)})${WILDCARD}*?`, {
    cardinality: negate2.cardinality,
    lookahead: negate2.lookahead
  });
}
function characterRangeToRegex(range2) {
  if (range2.right) {
    return withCardinality(`[${keywordToRegex(range2.left)}-${keywordToRegex(range2.right)}]`, {
      cardinality: range2.cardinality,
      lookahead: range2.lookahead,
      wrap: false
    });
  }
  return withCardinality(keywordToRegex(range2.left), {
    cardinality: range2.cardinality,
    lookahead: range2.lookahead,
    wrap: false
  });
}
function keywordToRegex(keyword) {
  return escapeRegExp(keyword.value);
}
function withCardinality(regex, options) {
  var _a;
  if (options.wrap !== false || options.lookahead) {
    regex = `(${(_a = options.lookahead) !== null && _a !== void 0 ? _a : ""}${regex})`;
  }
  if (options.cardinality) {
    return `${regex}${options.cardinality}`;
  }
  return regex;
}

// node_modules/langium/lib/languages/grammar-config.js
function createGrammarConfig(services) {
  const rules = [];
  const grammar = services.Grammar;
  for (const rule of grammar.rules) {
    if (isTerminalRule(rule) && isCommentTerminal(rule) && isMultilineComment(terminalRegex(rule))) {
      rules.push(rule.name);
    }
  }
  return {
    multilineCommentRules: rules,
    nameRegexp: DefaultNameRegexp
  };
}

// node_modules/lodash-es/_freeGlobal.js
var freeGlobal = typeof global == "object" && global && global.Object === Object && global;
var freeGlobal_default = freeGlobal;

// node_modules/lodash-es/_root.js
var freeSelf = typeof self == "object" && self && self.Object === Object && self;
var root = freeGlobal_default || freeSelf || Function("return this")();
var root_default = root;

// node_modules/lodash-es/_Symbol.js
var Symbol2 = root_default.Symbol;
var Symbol_default = Symbol2;

// node_modules/lodash-es/_getRawTag.js
var objectProto = Object.prototype;
var hasOwnProperty = objectProto.hasOwnProperty;
var nativeObjectToString = objectProto.toString;
var symToStringTag = Symbol_default ? Symbol_default.toStringTag : void 0;
function getRawTag(value) {
  var isOwn = hasOwnProperty.call(value, symToStringTag), tag = value[symToStringTag];
  try {
    value[symToStringTag] = void 0;
    var unmasked = true;
  } catch (e) {
  }
  var result2 = nativeObjectToString.call(value);
  if (unmasked) {
    if (isOwn) {
      value[symToStringTag] = tag;
    } else {
      delete value[symToStringTag];
    }
  }
  return result2;
}
var getRawTag_default = getRawTag;

// node_modules/lodash-es/_objectToString.js
var objectProto2 = Object.prototype;
var nativeObjectToString2 = objectProto2.toString;
function objectToString(value) {
  return nativeObjectToString2.call(value);
}
var objectToString_default = objectToString;

// node_modules/lodash-es/_baseGetTag.js
var nullTag = "[object Null]";
var undefinedTag = "[object Undefined]";
var symToStringTag2 = Symbol_default ? Symbol_default.toStringTag : void 0;
function baseGetTag(value) {
  if (value == null) {
    return value === void 0 ? undefinedTag : nullTag;
  }
  return symToStringTag2 && symToStringTag2 in Object(value) ? getRawTag_default(value) : objectToString_default(value);
}
var baseGetTag_default = baseGetTag;

// node_modules/lodash-es/isObjectLike.js
function isObjectLike(value) {
  return value != null && typeof value == "object";
}
var isObjectLike_default = isObjectLike;

// node_modules/lodash-es/isSymbol.js
var symbolTag = "[object Symbol]";
function isSymbol(value) {
  return typeof value == "symbol" || isObjectLike_default(value) && baseGetTag_default(value) == symbolTag;
}
var isSymbol_default = isSymbol;

// node_modules/lodash-es/_baseToNumber.js
var NAN = 0 / 0;
function baseToNumber(value) {
  if (typeof value == "number") {
    return value;
  }
  if (isSymbol_default(value)) {
    return NAN;
  }
  return +value;
}
var baseToNumber_default = baseToNumber;

// node_modules/lodash-es/_arrayMap.js
function arrayMap(array, iteratee2) {
  var index = -1, length = array == null ? 0 : array.length, result2 = Array(length);
  while (++index < length) {
    result2[index] = iteratee2(array[index], index, array);
  }
  return result2;
}
var arrayMap_default = arrayMap;

// node_modules/lodash-es/isArray.js
var isArray = Array.isArray;
var isArray_default = isArray;

// node_modules/lodash-es/_baseToString.js
var INFINITY = 1 / 0;
var symbolProto = Symbol_default ? Symbol_default.prototype : void 0;
var symbolToString = symbolProto ? symbolProto.toString : void 0;
function baseToString(value) {
  if (typeof value == "string") {
    return value;
  }
  if (isArray_default(value)) {
    return arrayMap_default(value, baseToString) + "";
  }
  if (isSymbol_default(value)) {
    return symbolToString ? symbolToString.call(value) : "";
  }
  var result2 = value + "";
  return result2 == "0" && 1 / value == -INFINITY ? "-0" : result2;
}
var baseToString_default = baseToString;

// node_modules/lodash-es/_createMathOperation.js
function createMathOperation(operator, defaultValue) {
  return function(value, other) {
    var result2;
    if (value === void 0 && other === void 0) {
      return defaultValue;
    }
    if (value !== void 0) {
      result2 = value;
    }
    if (other !== void 0) {
      if (result2 === void 0) {
        return other;
      }
      if (typeof value == "string" || typeof other == "string") {
        value = baseToString_default(value);
        other = baseToString_default(other);
      } else {
        value = baseToNumber_default(value);
        other = baseToNumber_default(other);
      }
      result2 = operator(value, other);
    }
    return result2;
  };
}
var createMathOperation_default = createMathOperation;

// node_modules/lodash-es/add.js
var add = createMathOperation_default(function(augend, addend) {
  return augend + addend;
}, 0);
var add_default = add;

// node_modules/lodash-es/_trimmedEndIndex.js
var reWhitespace = /\s/;
function trimmedEndIndex(string) {
  var index = string.length;
  while (index-- && reWhitespace.test(string.charAt(index))) {
  }
  return index;
}
var trimmedEndIndex_default = trimmedEndIndex;

// node_modules/lodash-es/_baseTrim.js
var reTrimStart = /^\s+/;
function baseTrim(string) {
  return string ? string.slice(0, trimmedEndIndex_default(string) + 1).replace(reTrimStart, "") : string;
}
var baseTrim_default = baseTrim;

// node_modules/lodash-es/isObject.js
function isObject(value) {
  var type = typeof value;
  return value != null && (type == "object" || type == "function");
}
var isObject_default = isObject;

// node_modules/lodash-es/toNumber.js
var NAN2 = 0 / 0;
var reIsBadHex = /^[-+]0x[0-9a-f]+$/i;
var reIsBinary = /^0b[01]+$/i;
var reIsOctal = /^0o[0-7]+$/i;
var freeParseInt = parseInt;
function toNumber(value) {
  if (typeof value == "number") {
    return value;
  }
  if (isSymbol_default(value)) {
    return NAN2;
  }
  if (isObject_default(value)) {
    var other = typeof value.valueOf == "function" ? value.valueOf() : value;
    value = isObject_default(other) ? other + "" : other;
  }
  if (typeof value != "string") {
    return value === 0 ? value : +value;
  }
  value = baseTrim_default(value);
  var isBinary = reIsBinary.test(value);
  return isBinary || reIsOctal.test(value) ? freeParseInt(value.slice(2), isBinary ? 2 : 8) : reIsBadHex.test(value) ? NAN2 : +value;
}
var toNumber_default = toNumber;

// node_modules/lodash-es/toFinite.js
var INFINITY2 = 1 / 0;
var MAX_INTEGER = 17976931348623157e292;
function toFinite(value) {
  if (!value) {
    return value === 0 ? value : 0;
  }
  value = toNumber_default(value);
  if (value === INFINITY2 || value === -INFINITY2) {
    var sign = value < 0 ? -1 : 1;
    return sign * MAX_INTEGER;
  }
  return value === value ? value : 0;
}
var toFinite_default = toFinite;

// node_modules/lodash-es/toInteger.js
function toInteger(value) {
  var result2 = toFinite_default(value), remainder = result2 % 1;
  return result2 === result2 ? remainder ? result2 - remainder : result2 : 0;
}
var toInteger_default = toInteger;

// node_modules/lodash-es/after.js
var FUNC_ERROR_TEXT = "Expected a function";
function after(n, func) {
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT);
  }
  n = toInteger_default(n);
  return function() {
    if (--n < 1) {
      return func.apply(this, arguments);
    }
  };
}
var after_default = after;

// node_modules/lodash-es/identity.js
function identity(value) {
  return value;
}
var identity_default = identity;

// node_modules/lodash-es/isFunction.js
var asyncTag = "[object AsyncFunction]";
var funcTag = "[object Function]";
var genTag = "[object GeneratorFunction]";
var proxyTag = "[object Proxy]";
function isFunction(value) {
  if (!isObject_default(value)) {
    return false;
  }
  var tag = baseGetTag_default(value);
  return tag == funcTag || tag == genTag || tag == asyncTag || tag == proxyTag;
}
var isFunction_default = isFunction;

// node_modules/lodash-es/_coreJsData.js
var coreJsData = root_default["__core-js_shared__"];
var coreJsData_default = coreJsData;

// node_modules/lodash-es/_isMasked.js
var maskSrcKey = function() {
  var uid = /[^.]+$/.exec(coreJsData_default && coreJsData_default.keys && coreJsData_default.keys.IE_PROTO || "");
  return uid ? "Symbol(src)_1." + uid : "";
}();
function isMasked(func) {
  return !!maskSrcKey && maskSrcKey in func;
}
var isMasked_default = isMasked;

// node_modules/lodash-es/_toSource.js
var funcProto = Function.prototype;
var funcToString = funcProto.toString;
function toSource(func) {
  if (func != null) {
    try {
      return funcToString.call(func);
    } catch (e) {
    }
    try {
      return func + "";
    } catch (e) {
    }
  }
  return "";
}
var toSource_default = toSource;

// node_modules/lodash-es/_baseIsNative.js
var reRegExpChar = /[\\^$.*+?()[\]{}|]/g;
var reIsHostCtor = /^\[object .+?Constructor\]$/;
var funcProto2 = Function.prototype;
var objectProto3 = Object.prototype;
var funcToString2 = funcProto2.toString;
var hasOwnProperty2 = objectProto3.hasOwnProperty;
var reIsNative = RegExp(
  "^" + funcToString2.call(hasOwnProperty2).replace(reRegExpChar, "\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, "$1.*?") + "$"
);
function baseIsNative(value) {
  if (!isObject_default(value) || isMasked_default(value)) {
    return false;
  }
  var pattern = isFunction_default(value) ? reIsNative : reIsHostCtor;
  return pattern.test(toSource_default(value));
}
var baseIsNative_default = baseIsNative;

// node_modules/lodash-es/_getValue.js
function getValue(object, key) {
  return object == null ? void 0 : object[key];
}
var getValue_default = getValue;

// node_modules/lodash-es/_getNative.js
function getNative(object, key) {
  var value = getValue_default(object, key);
  return baseIsNative_default(value) ? value : void 0;
}
var getNative_default = getNative;

// node_modules/lodash-es/_WeakMap.js
var WeakMap = getNative_default(root_default, "WeakMap");
var WeakMap_default = WeakMap;

// node_modules/lodash-es/_metaMap.js
var metaMap = WeakMap_default && new WeakMap_default();
var metaMap_default = metaMap;

// node_modules/lodash-es/_baseSetData.js
var baseSetData = !metaMap_default ? identity_default : function(func, data) {
  metaMap_default.set(func, data);
  return func;
};
var baseSetData_default = baseSetData;

// node_modules/lodash-es/_baseCreate.js
var objectCreate = Object.create;
var baseCreate = /* @__PURE__ */ function() {
  function object() {
  }
  return function(proto) {
    if (!isObject_default(proto)) {
      return {};
    }
    if (objectCreate) {
      return objectCreate(proto);
    }
    object.prototype = proto;
    var result2 = new object();
    object.prototype = void 0;
    return result2;
  };
}();
var baseCreate_default = baseCreate;

// node_modules/lodash-es/_createCtor.js
function createCtor(Ctor) {
  return function() {
    var args = arguments;
    switch (args.length) {
      case 0:
        return new Ctor();
      case 1:
        return new Ctor(args[0]);
      case 2:
        return new Ctor(args[0], args[1]);
      case 3:
        return new Ctor(args[0], args[1], args[2]);
      case 4:
        return new Ctor(args[0], args[1], args[2], args[3]);
      case 5:
        return new Ctor(args[0], args[1], args[2], args[3], args[4]);
      case 6:
        return new Ctor(args[0], args[1], args[2], args[3], args[4], args[5]);
      case 7:
        return new Ctor(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    }
    var thisBinding = baseCreate_default(Ctor.prototype), result2 = Ctor.apply(thisBinding, args);
    return isObject_default(result2) ? result2 : thisBinding;
  };
}
var createCtor_default = createCtor;

// node_modules/lodash-es/_createBind.js
var WRAP_BIND_FLAG = 1;
function createBind(func, bitmask, thisArg) {
  var isBind = bitmask & WRAP_BIND_FLAG, Ctor = createCtor_default(func);
  function wrapper() {
    var fn = this && this !== root_default && this instanceof wrapper ? Ctor : func;
    return fn.apply(isBind ? thisArg : this, arguments);
  }
  return wrapper;
}
var createBind_default = createBind;

// node_modules/lodash-es/_apply.js
function apply(func, thisArg, args) {
  switch (args.length) {
    case 0:
      return func.call(thisArg);
    case 1:
      return func.call(thisArg, args[0]);
    case 2:
      return func.call(thisArg, args[0], args[1]);
    case 3:
      return func.call(thisArg, args[0], args[1], args[2]);
  }
  return func.apply(thisArg, args);
}
var apply_default = apply;

// node_modules/lodash-es/_composeArgs.js
var nativeMax = Math.max;
function composeArgs(args, partials, holders, isCurried) {
  var argsIndex = -1, argsLength = args.length, holdersLength = holders.length, leftIndex = -1, leftLength = partials.length, rangeLength = nativeMax(argsLength - holdersLength, 0), result2 = Array(leftLength + rangeLength), isUncurried = !isCurried;
  while (++leftIndex < leftLength) {
    result2[leftIndex] = partials[leftIndex];
  }
  while (++argsIndex < holdersLength) {
    if (isUncurried || argsIndex < argsLength) {
      result2[holders[argsIndex]] = args[argsIndex];
    }
  }
  while (rangeLength--) {
    result2[leftIndex++] = args[argsIndex++];
  }
  return result2;
}
var composeArgs_default = composeArgs;

// node_modules/lodash-es/_composeArgsRight.js
var nativeMax2 = Math.max;
function composeArgsRight(args, partials, holders, isCurried) {
  var argsIndex = -1, argsLength = args.length, holdersIndex = -1, holdersLength = holders.length, rightIndex = -1, rightLength = partials.length, rangeLength = nativeMax2(argsLength - holdersLength, 0), result2 = Array(rangeLength + rightLength), isUncurried = !isCurried;
  while (++argsIndex < rangeLength) {
    result2[argsIndex] = args[argsIndex];
  }
  var offset = argsIndex;
  while (++rightIndex < rightLength) {
    result2[offset + rightIndex] = partials[rightIndex];
  }
  while (++holdersIndex < holdersLength) {
    if (isUncurried || argsIndex < argsLength) {
      result2[offset + holders[holdersIndex]] = args[argsIndex++];
    }
  }
  return result2;
}
var composeArgsRight_default = composeArgsRight;

// node_modules/lodash-es/_countHolders.js
function countHolders(array, placeholder) {
  var length = array.length, result2 = 0;
  while (length--) {
    if (array[length] === placeholder) {
      ++result2;
    }
  }
  return result2;
}
var countHolders_default = countHolders;

// node_modules/lodash-es/_baseLodash.js
function baseLodash() {
}
var baseLodash_default = baseLodash;

// node_modules/lodash-es/_LazyWrapper.js
var MAX_ARRAY_LENGTH = 4294967295;
function LazyWrapper(value) {
  this.__wrapped__ = value;
  this.__actions__ = [];
  this.__dir__ = 1;
  this.__filtered__ = false;
  this.__iteratees__ = [];
  this.__takeCount__ = MAX_ARRAY_LENGTH;
  this.__views__ = [];
}
LazyWrapper.prototype = baseCreate_default(baseLodash_default.prototype);
LazyWrapper.prototype.constructor = LazyWrapper;
var LazyWrapper_default = LazyWrapper;

// node_modules/lodash-es/noop.js
function noop() {
}
var noop_default = noop;

// node_modules/lodash-es/_getData.js
var getData = !metaMap_default ? noop_default : function(func) {
  return metaMap_default.get(func);
};
var getData_default = getData;

// node_modules/lodash-es/_realNames.js
var realNames = {};
var realNames_default = realNames;

// node_modules/lodash-es/_getFuncName.js
var objectProto4 = Object.prototype;
var hasOwnProperty3 = objectProto4.hasOwnProperty;
function getFuncName(func) {
  var result2 = func.name + "", array = realNames_default[result2], length = hasOwnProperty3.call(realNames_default, result2) ? array.length : 0;
  while (length--) {
    var data = array[length], otherFunc = data.func;
    if (otherFunc == null || otherFunc == func) {
      return data.name;
    }
  }
  return result2;
}
var getFuncName_default = getFuncName;

// node_modules/lodash-es/_LodashWrapper.js
function LodashWrapper(value, chainAll) {
  this.__wrapped__ = value;
  this.__actions__ = [];
  this.__chain__ = !!chainAll;
  this.__index__ = 0;
  this.__values__ = void 0;
}
LodashWrapper.prototype = baseCreate_default(baseLodash_default.prototype);
LodashWrapper.prototype.constructor = LodashWrapper;
var LodashWrapper_default = LodashWrapper;

// node_modules/lodash-es/_copyArray.js
function copyArray(source, array) {
  var index = -1, length = source.length;
  array || (array = Array(length));
  while (++index < length) {
    array[index] = source[index];
  }
  return array;
}
var copyArray_default = copyArray;

// node_modules/lodash-es/_wrapperClone.js
function wrapperClone(wrapper) {
  if (wrapper instanceof LazyWrapper_default) {
    return wrapper.clone();
  }
  var result2 = new LodashWrapper_default(wrapper.__wrapped__, wrapper.__chain__);
  result2.__actions__ = copyArray_default(wrapper.__actions__);
  result2.__index__ = wrapper.__index__;
  result2.__values__ = wrapper.__values__;
  return result2;
}
var wrapperClone_default = wrapperClone;

// node_modules/lodash-es/wrapperLodash.js
var objectProto5 = Object.prototype;
var hasOwnProperty4 = objectProto5.hasOwnProperty;
function lodash(value) {
  if (isObjectLike_default(value) && !isArray_default(value) && !(value instanceof LazyWrapper_default)) {
    if (value instanceof LodashWrapper_default) {
      return value;
    }
    if (hasOwnProperty4.call(value, "__wrapped__")) {
      return wrapperClone_default(value);
    }
  }
  return new LodashWrapper_default(value);
}
lodash.prototype = baseLodash_default.prototype;
lodash.prototype.constructor = lodash;
var wrapperLodash_default = lodash;

// node_modules/lodash-es/_isLaziable.js
function isLaziable(func) {
  var funcName = getFuncName_default(func), other = wrapperLodash_default[funcName];
  if (typeof other != "function" || !(funcName in LazyWrapper_default.prototype)) {
    return false;
  }
  if (func === other) {
    return true;
  }
  var data = getData_default(other);
  return !!data && func === data[0];
}
var isLaziable_default = isLaziable;

// node_modules/lodash-es/_shortOut.js
var HOT_COUNT = 800;
var HOT_SPAN = 16;
var nativeNow = Date.now;
function shortOut(func) {
  var count = 0, lastCalled = 0;
  return function() {
    var stamp = nativeNow(), remaining = HOT_SPAN - (stamp - lastCalled);
    lastCalled = stamp;
    if (remaining > 0) {
      if (++count >= HOT_COUNT) {
        return arguments[0];
      }
    } else {
      count = 0;
    }
    return func.apply(void 0, arguments);
  };
}
var shortOut_default = shortOut;

// node_modules/lodash-es/_setData.js
var setData = shortOut_default(baseSetData_default);
var setData_default = setData;

// node_modules/lodash-es/_getWrapDetails.js
var reWrapDetails = /\{\n\/\* \[wrapped with (.+)\] \*/;
var reSplitDetails = /,? & /;
function getWrapDetails(source) {
  var match = source.match(reWrapDetails);
  return match ? match[1].split(reSplitDetails) : [];
}
var getWrapDetails_default = getWrapDetails;

// node_modules/lodash-es/_insertWrapDetails.js
var reWrapComment = /\{(?:\n\/\* \[wrapped with .+\] \*\/)?\n?/;
function insertWrapDetails(source, details) {
  var length = details.length;
  if (!length) {
    return source;
  }
  var lastIndex = length - 1;
  details[lastIndex] = (length > 1 ? "& " : "") + details[lastIndex];
  details = details.join(length > 2 ? ", " : " ");
  return source.replace(reWrapComment, "{\n/* [wrapped with " + details + "] */\n");
}
var insertWrapDetails_default = insertWrapDetails;

// node_modules/lodash-es/constant.js
function constant(value) {
  return function() {
    return value;
  };
}
var constant_default = constant;

// node_modules/lodash-es/_defineProperty.js
var defineProperty = function() {
  try {
    var func = getNative_default(Object, "defineProperty");
    func({}, "", {});
    return func;
  } catch (e) {
  }
}();
var defineProperty_default = defineProperty;

// node_modules/lodash-es/_baseSetToString.js
var baseSetToString = !defineProperty_default ? identity_default : function(func, string) {
  return defineProperty_default(func, "toString", {
    "configurable": true,
    "enumerable": false,
    "value": constant_default(string),
    "writable": true
  });
};
var baseSetToString_default = baseSetToString;

// node_modules/lodash-es/_setToString.js
var setToString = shortOut_default(baseSetToString_default);
var setToString_default = setToString;

// node_modules/lodash-es/_arrayEach.js
function arrayEach(array, iteratee2) {
  var index = -1, length = array == null ? 0 : array.length;
  while (++index < length) {
    if (iteratee2(array[index], index, array) === false) {
      break;
    }
  }
  return array;
}
var arrayEach_default = arrayEach;

// node_modules/lodash-es/_baseFindIndex.js
function baseFindIndex(array, predicate, fromIndex, fromRight) {
  var length = array.length, index = fromIndex + (fromRight ? 1 : -1);
  while (fromRight ? index-- : ++index < length) {
    if (predicate(array[index], index, array)) {
      return index;
    }
  }
  return -1;
}
var baseFindIndex_default = baseFindIndex;

// node_modules/lodash-es/_baseIsNaN.js
function baseIsNaN(value) {
  return value !== value;
}
var baseIsNaN_default = baseIsNaN;

// node_modules/lodash-es/_strictIndexOf.js
function strictIndexOf(array, value, fromIndex) {
  var index = fromIndex - 1, length = array.length;
  while (++index < length) {
    if (array[index] === value) {
      return index;
    }
  }
  return -1;
}
var strictIndexOf_default = strictIndexOf;

// node_modules/lodash-es/_baseIndexOf.js
function baseIndexOf(array, value, fromIndex) {
  return value === value ? strictIndexOf_default(array, value, fromIndex) : baseFindIndex_default(array, baseIsNaN_default, fromIndex);
}
var baseIndexOf_default = baseIndexOf;

// node_modules/lodash-es/_arrayIncludes.js
function arrayIncludes(array, value) {
  var length = array == null ? 0 : array.length;
  return !!length && baseIndexOf_default(array, value, 0) > -1;
}
var arrayIncludes_default = arrayIncludes;

// node_modules/lodash-es/_updateWrapDetails.js
var WRAP_BIND_FLAG2 = 1;
var WRAP_BIND_KEY_FLAG = 2;
var WRAP_CURRY_FLAG = 8;
var WRAP_CURRY_RIGHT_FLAG = 16;
var WRAP_PARTIAL_FLAG = 32;
var WRAP_PARTIAL_RIGHT_FLAG = 64;
var WRAP_ARY_FLAG = 128;
var WRAP_REARG_FLAG = 256;
var WRAP_FLIP_FLAG = 512;
var wrapFlags = [
  ["ary", WRAP_ARY_FLAG],
  ["bind", WRAP_BIND_FLAG2],
  ["bindKey", WRAP_BIND_KEY_FLAG],
  ["curry", WRAP_CURRY_FLAG],
  ["curryRight", WRAP_CURRY_RIGHT_FLAG],
  ["flip", WRAP_FLIP_FLAG],
  ["partial", WRAP_PARTIAL_FLAG],
  ["partialRight", WRAP_PARTIAL_RIGHT_FLAG],
  ["rearg", WRAP_REARG_FLAG]
];
function updateWrapDetails(details, bitmask) {
  arrayEach_default(wrapFlags, function(pair) {
    var value = "_." + pair[0];
    if (bitmask & pair[1] && !arrayIncludes_default(details, value)) {
      details.push(value);
    }
  });
  return details.sort();
}
var updateWrapDetails_default = updateWrapDetails;

// node_modules/lodash-es/_setWrapToString.js
function setWrapToString(wrapper, reference, bitmask) {
  var source = reference + "";
  return setToString_default(wrapper, insertWrapDetails_default(source, updateWrapDetails_default(getWrapDetails_default(source), bitmask)));
}
var setWrapToString_default = setWrapToString;

// node_modules/lodash-es/_createRecurry.js
var WRAP_BIND_FLAG3 = 1;
var WRAP_BIND_KEY_FLAG2 = 2;
var WRAP_CURRY_BOUND_FLAG = 4;
var WRAP_CURRY_FLAG2 = 8;
var WRAP_PARTIAL_FLAG2 = 32;
var WRAP_PARTIAL_RIGHT_FLAG2 = 64;
function createRecurry(func, bitmask, wrapFunc, placeholder, thisArg, partials, holders, argPos, ary2, arity) {
  var isCurry = bitmask & WRAP_CURRY_FLAG2, newHolders = isCurry ? holders : void 0, newHoldersRight = isCurry ? void 0 : holders, newPartials = isCurry ? partials : void 0, newPartialsRight = isCurry ? void 0 : partials;
  bitmask |= isCurry ? WRAP_PARTIAL_FLAG2 : WRAP_PARTIAL_RIGHT_FLAG2;
  bitmask &= ~(isCurry ? WRAP_PARTIAL_RIGHT_FLAG2 : WRAP_PARTIAL_FLAG2);
  if (!(bitmask & WRAP_CURRY_BOUND_FLAG)) {
    bitmask &= ~(WRAP_BIND_FLAG3 | WRAP_BIND_KEY_FLAG2);
  }
  var newData = [
    func,
    bitmask,
    thisArg,
    newPartials,
    newHolders,
    newPartialsRight,
    newHoldersRight,
    argPos,
    ary2,
    arity
  ];
  var result2 = wrapFunc.apply(void 0, newData);
  if (isLaziable_default(func)) {
    setData_default(result2, newData);
  }
  result2.placeholder = placeholder;
  return setWrapToString_default(result2, func, bitmask);
}
var createRecurry_default = createRecurry;

// node_modules/lodash-es/_getHolder.js
function getHolder(func) {
  var object = func;
  return object.placeholder;
}
var getHolder_default = getHolder;

// node_modules/lodash-es/_isIndex.js
var MAX_SAFE_INTEGER = 9007199254740991;
var reIsUint = /^(?:0|[1-9]\d*)$/;
function isIndex(value, length) {
  var type = typeof value;
  length = length == null ? MAX_SAFE_INTEGER : length;
  return !!length && (type == "number" || type != "symbol" && reIsUint.test(value)) && (value > -1 && value % 1 == 0 && value < length);
}
var isIndex_default = isIndex;

// node_modules/lodash-es/_reorder.js
var nativeMin = Math.min;
function reorder(array, indexes) {
  var arrLength = array.length, length = nativeMin(indexes.length, arrLength), oldArray = copyArray_default(array);
  while (length--) {
    var index = indexes[length];
    array[length] = isIndex_default(index, arrLength) ? oldArray[index] : void 0;
  }
  return array;
}
var reorder_default = reorder;

// node_modules/lodash-es/_replaceHolders.js
var PLACEHOLDER = "__lodash_placeholder__";
function replaceHolders(array, placeholder) {
  var index = -1, length = array.length, resIndex = 0, result2 = [];
  while (++index < length) {
    var value = array[index];
    if (value === placeholder || value === PLACEHOLDER) {
      array[index] = PLACEHOLDER;
      result2[resIndex++] = index;
    }
  }
  return result2;
}
var replaceHolders_default = replaceHolders;

// node_modules/lodash-es/_createHybrid.js
var WRAP_BIND_FLAG4 = 1;
var WRAP_BIND_KEY_FLAG3 = 2;
var WRAP_CURRY_FLAG3 = 8;
var WRAP_CURRY_RIGHT_FLAG2 = 16;
var WRAP_ARY_FLAG2 = 128;
var WRAP_FLIP_FLAG2 = 512;
function createHybrid(func, bitmask, thisArg, partials, holders, partialsRight, holdersRight, argPos, ary2, arity) {
  var isAry = bitmask & WRAP_ARY_FLAG2, isBind = bitmask & WRAP_BIND_FLAG4, isBindKey = bitmask & WRAP_BIND_KEY_FLAG3, isCurried = bitmask & (WRAP_CURRY_FLAG3 | WRAP_CURRY_RIGHT_FLAG2), isFlip = bitmask & WRAP_FLIP_FLAG2, Ctor = isBindKey ? void 0 : createCtor_default(func);
  function wrapper() {
    var length = arguments.length, args = Array(length), index = length;
    while (index--) {
      args[index] = arguments[index];
    }
    if (isCurried) {
      var placeholder = getHolder_default(wrapper), holdersCount = countHolders_default(args, placeholder);
    }
    if (partials) {
      args = composeArgs_default(args, partials, holders, isCurried);
    }
    if (partialsRight) {
      args = composeArgsRight_default(args, partialsRight, holdersRight, isCurried);
    }
    length -= holdersCount;
    if (isCurried && length < arity) {
      var newHolders = replaceHolders_default(args, placeholder);
      return createRecurry_default(
        func,
        bitmask,
        createHybrid,
        wrapper.placeholder,
        thisArg,
        args,
        newHolders,
        argPos,
        ary2,
        arity - length
      );
    }
    var thisBinding = isBind ? thisArg : this, fn = isBindKey ? thisBinding[func] : func;
    length = args.length;
    if (argPos) {
      args = reorder_default(args, argPos);
    } else if (isFlip && length > 1) {
      args.reverse();
    }
    if (isAry && ary2 < length) {
      args.length = ary2;
    }
    if (this && this !== root_default && this instanceof wrapper) {
      fn = Ctor || createCtor_default(fn);
    }
    return fn.apply(thisBinding, args);
  }
  return wrapper;
}
var createHybrid_default = createHybrid;

// node_modules/lodash-es/_createCurry.js
function createCurry(func, bitmask, arity) {
  var Ctor = createCtor_default(func);
  function wrapper() {
    var length = arguments.length, args = Array(length), index = length, placeholder = getHolder_default(wrapper);
    while (index--) {
      args[index] = arguments[index];
    }
    var holders = length < 3 && args[0] !== placeholder && args[length - 1] !== placeholder ? [] : replaceHolders_default(args, placeholder);
    length -= holders.length;
    if (length < arity) {
      return createRecurry_default(
        func,
        bitmask,
        createHybrid_default,
        wrapper.placeholder,
        void 0,
        args,
        holders,
        void 0,
        void 0,
        arity - length
      );
    }
    var fn = this && this !== root_default && this instanceof wrapper ? Ctor : func;
    return apply_default(fn, this, args);
  }
  return wrapper;
}
var createCurry_default = createCurry;

// node_modules/lodash-es/_createPartial.js
var WRAP_BIND_FLAG5 = 1;
function createPartial(func, bitmask, thisArg, partials) {
  var isBind = bitmask & WRAP_BIND_FLAG5, Ctor = createCtor_default(func);
  function wrapper() {
    var argsIndex = -1, argsLength = arguments.length, leftIndex = -1, leftLength = partials.length, args = Array(leftLength + argsLength), fn = this && this !== root_default && this instanceof wrapper ? Ctor : func;
    while (++leftIndex < leftLength) {
      args[leftIndex] = partials[leftIndex];
    }
    while (argsLength--) {
      args[leftIndex++] = arguments[++argsIndex];
    }
    return apply_default(fn, isBind ? thisArg : this, args);
  }
  return wrapper;
}
var createPartial_default = createPartial;

// node_modules/lodash-es/_mergeData.js
var PLACEHOLDER2 = "__lodash_placeholder__";
var WRAP_BIND_FLAG6 = 1;
var WRAP_BIND_KEY_FLAG4 = 2;
var WRAP_CURRY_BOUND_FLAG2 = 4;
var WRAP_CURRY_FLAG4 = 8;
var WRAP_ARY_FLAG3 = 128;
var WRAP_REARG_FLAG2 = 256;
var nativeMin2 = Math.min;
function mergeData(data, source) {
  var bitmask = data[1], srcBitmask = source[1], newBitmask = bitmask | srcBitmask, isCommon = newBitmask < (WRAP_BIND_FLAG6 | WRAP_BIND_KEY_FLAG4 | WRAP_ARY_FLAG3);
  var isCombo = srcBitmask == WRAP_ARY_FLAG3 && bitmask == WRAP_CURRY_FLAG4 || srcBitmask == WRAP_ARY_FLAG3 && bitmask == WRAP_REARG_FLAG2 && data[7].length <= source[8] || srcBitmask == (WRAP_ARY_FLAG3 | WRAP_REARG_FLAG2) && source[7].length <= source[8] && bitmask == WRAP_CURRY_FLAG4;
  if (!(isCommon || isCombo)) {
    return data;
  }
  if (srcBitmask & WRAP_BIND_FLAG6) {
    data[2] = source[2];
    newBitmask |= bitmask & WRAP_BIND_FLAG6 ? 0 : WRAP_CURRY_BOUND_FLAG2;
  }
  var value = source[3];
  if (value) {
    var partials = data[3];
    data[3] = partials ? composeArgs_default(partials, value, source[4]) : value;
    data[4] = partials ? replaceHolders_default(data[3], PLACEHOLDER2) : source[4];
  }
  value = source[5];
  if (value) {
    partials = data[5];
    data[5] = partials ? composeArgsRight_default(partials, value, source[6]) : value;
    data[6] = partials ? replaceHolders_default(data[5], PLACEHOLDER2) : source[6];
  }
  value = source[7];
  if (value) {
    data[7] = value;
  }
  if (srcBitmask & WRAP_ARY_FLAG3) {
    data[8] = data[8] == null ? source[8] : nativeMin2(data[8], source[8]);
  }
  if (data[9] == null) {
    data[9] = source[9];
  }
  data[0] = source[0];
  data[1] = newBitmask;
  return data;
}
var mergeData_default = mergeData;

// node_modules/lodash-es/_createWrap.js
var FUNC_ERROR_TEXT2 = "Expected a function";
var WRAP_BIND_FLAG7 = 1;
var WRAP_BIND_KEY_FLAG5 = 2;
var WRAP_CURRY_FLAG5 = 8;
var WRAP_CURRY_RIGHT_FLAG3 = 16;
var WRAP_PARTIAL_FLAG3 = 32;
var WRAP_PARTIAL_RIGHT_FLAG3 = 64;
var nativeMax3 = Math.max;
function createWrap(func, bitmask, thisArg, partials, holders, argPos, ary2, arity) {
  var isBindKey = bitmask & WRAP_BIND_KEY_FLAG5;
  if (!isBindKey && typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT2);
  }
  var length = partials ? partials.length : 0;
  if (!length) {
    bitmask &= ~(WRAP_PARTIAL_FLAG3 | WRAP_PARTIAL_RIGHT_FLAG3);
    partials = holders = void 0;
  }
  ary2 = ary2 === void 0 ? ary2 : nativeMax3(toInteger_default(ary2), 0);
  arity = arity === void 0 ? arity : toInteger_default(arity);
  length -= holders ? holders.length : 0;
  if (bitmask & WRAP_PARTIAL_RIGHT_FLAG3) {
    var partialsRight = partials, holdersRight = holders;
    partials = holders = void 0;
  }
  var data = isBindKey ? void 0 : getData_default(func);
  var newData = [
    func,
    bitmask,
    thisArg,
    partials,
    holders,
    partialsRight,
    holdersRight,
    argPos,
    ary2,
    arity
  ];
  if (data) {
    mergeData_default(newData, data);
  }
  func = newData[0];
  bitmask = newData[1];
  thisArg = newData[2];
  partials = newData[3];
  holders = newData[4];
  arity = newData[9] = newData[9] === void 0 ? isBindKey ? 0 : func.length : nativeMax3(newData[9] - length, 0);
  if (!arity && bitmask & (WRAP_CURRY_FLAG5 | WRAP_CURRY_RIGHT_FLAG3)) {
    bitmask &= ~(WRAP_CURRY_FLAG5 | WRAP_CURRY_RIGHT_FLAG3);
  }
  if (!bitmask || bitmask == WRAP_BIND_FLAG7) {
    var result2 = createBind_default(func, bitmask, thisArg);
  } else if (bitmask == WRAP_CURRY_FLAG5 || bitmask == WRAP_CURRY_RIGHT_FLAG3) {
    result2 = createCurry_default(func, bitmask, arity);
  } else if ((bitmask == WRAP_PARTIAL_FLAG3 || bitmask == (WRAP_BIND_FLAG7 | WRAP_PARTIAL_FLAG3)) && !holders.length) {
    result2 = createPartial_default(func, bitmask, thisArg, partials);
  } else {
    result2 = createHybrid_default.apply(void 0, newData);
  }
  var setter = data ? baseSetData_default : setData_default;
  return setWrapToString_default(setter(result2, newData), func, bitmask);
}
var createWrap_default = createWrap;

// node_modules/lodash-es/ary.js
var WRAP_ARY_FLAG4 = 128;
function ary(func, n, guard) {
  n = guard ? void 0 : n;
  n = func && n == null ? func.length : n;
  return createWrap_default(func, WRAP_ARY_FLAG4, void 0, void 0, void 0, void 0, n);
}
var ary_default = ary;

// node_modules/lodash-es/_baseAssignValue.js
function baseAssignValue(object, key, value) {
  if (key == "__proto__" && defineProperty_default) {
    defineProperty_default(object, key, {
      "configurable": true,
      "enumerable": true,
      "value": value,
      "writable": true
    });
  } else {
    object[key] = value;
  }
}
var baseAssignValue_default = baseAssignValue;

// node_modules/lodash-es/eq.js
function eq(value, other) {
  return value === other || value !== value && other !== other;
}
var eq_default = eq;

// node_modules/lodash-es/_assignValue.js
var objectProto6 = Object.prototype;
var hasOwnProperty5 = objectProto6.hasOwnProperty;
function assignValue(object, key, value) {
  var objValue = object[key];
  if (!(hasOwnProperty5.call(object, key) && eq_default(objValue, value)) || value === void 0 && !(key in object)) {
    baseAssignValue_default(object, key, value);
  }
}
var assignValue_default = assignValue;

// node_modules/lodash-es/_copyObject.js
function copyObject(source, props, object, customizer) {
  var isNew = !object;
  object || (object = {});
  var index = -1, length = props.length;
  while (++index < length) {
    var key = props[index];
    var newValue = customizer ? customizer(object[key], source[key], key, object, source) : void 0;
    if (newValue === void 0) {
      newValue = source[key];
    }
    if (isNew) {
      baseAssignValue_default(object, key, newValue);
    } else {
      assignValue_default(object, key, newValue);
    }
  }
  return object;
}
var copyObject_default = copyObject;

// node_modules/lodash-es/_overRest.js
var nativeMax4 = Math.max;
function overRest(func, start, transform2) {
  start = nativeMax4(start === void 0 ? func.length - 1 : start, 0);
  return function() {
    var args = arguments, index = -1, length = nativeMax4(args.length - start, 0), array = Array(length);
    while (++index < length) {
      array[index] = args[start + index];
    }
    index = -1;
    var otherArgs = Array(start + 1);
    while (++index < start) {
      otherArgs[index] = args[index];
    }
    otherArgs[start] = transform2(array);
    return apply_default(func, this, otherArgs);
  };
}
var overRest_default = overRest;

// node_modules/lodash-es/_baseRest.js
function baseRest(func, start) {
  return setToString_default(overRest_default(func, start, identity_default), func + "");
}
var baseRest_default = baseRest;

// node_modules/lodash-es/isLength.js
var MAX_SAFE_INTEGER2 = 9007199254740991;
function isLength(value) {
  return typeof value == "number" && value > -1 && value % 1 == 0 && value <= MAX_SAFE_INTEGER2;
}
var isLength_default = isLength;

// node_modules/lodash-es/isArrayLike.js
function isArrayLike(value) {
  return value != null && isLength_default(value.length) && !isFunction_default(value);
}
var isArrayLike_default = isArrayLike;

// node_modules/lodash-es/_isIterateeCall.js
function isIterateeCall(value, index, object) {
  if (!isObject_default(object)) {
    return false;
  }
  var type = typeof index;
  if (type == "number" ? isArrayLike_default(object) && isIndex_default(index, object.length) : type == "string" && index in object) {
    return eq_default(object[index], value);
  }
  return false;
}
var isIterateeCall_default = isIterateeCall;

// node_modules/lodash-es/_createAssigner.js
function createAssigner(assigner) {
  return baseRest_default(function(object, sources) {
    var index = -1, length = sources.length, customizer = length > 1 ? sources[length - 1] : void 0, guard = length > 2 ? sources[2] : void 0;
    customizer = assigner.length > 3 && typeof customizer == "function" ? (length--, customizer) : void 0;
    if (guard && isIterateeCall_default(sources[0], sources[1], guard)) {
      customizer = length < 3 ? void 0 : customizer;
      length = 1;
    }
    object = Object(object);
    while (++index < length) {
      var source = sources[index];
      if (source) {
        assigner(object, source, index, customizer);
      }
    }
    return object;
  });
}
var createAssigner_default = createAssigner;

// node_modules/lodash-es/_isPrototype.js
var objectProto7 = Object.prototype;
function isPrototype(value) {
  var Ctor = value && value.constructor, proto = typeof Ctor == "function" && Ctor.prototype || objectProto7;
  return value === proto;
}
var isPrototype_default = isPrototype;

// node_modules/lodash-es/_baseTimes.js
function baseTimes(n, iteratee2) {
  var index = -1, result2 = Array(n);
  while (++index < n) {
    result2[index] = iteratee2(index);
  }
  return result2;
}
var baseTimes_default = baseTimes;

// node_modules/lodash-es/_baseIsArguments.js
var argsTag = "[object Arguments]";
function baseIsArguments(value) {
  return isObjectLike_default(value) && baseGetTag_default(value) == argsTag;
}
var baseIsArguments_default = baseIsArguments;

// node_modules/lodash-es/isArguments.js
var objectProto8 = Object.prototype;
var hasOwnProperty6 = objectProto8.hasOwnProperty;
var propertyIsEnumerable = objectProto8.propertyIsEnumerable;
var isArguments = baseIsArguments_default(/* @__PURE__ */ function() {
  return arguments;
}()) ? baseIsArguments_default : function(value) {
  return isObjectLike_default(value) && hasOwnProperty6.call(value, "callee") && !propertyIsEnumerable.call(value, "callee");
};
var isArguments_default = isArguments;

// node_modules/lodash-es/stubFalse.js
function stubFalse() {
  return false;
}
var stubFalse_default = stubFalse;

// node_modules/lodash-es/isBuffer.js
var freeExports = typeof exports == "object" && exports && !exports.nodeType && exports;
var freeModule = freeExports && typeof module == "object" && module && !module.nodeType && module;
var moduleExports = freeModule && freeModule.exports === freeExports;
var Buffer = moduleExports ? root_default.Buffer : void 0;
var nativeIsBuffer = Buffer ? Buffer.isBuffer : void 0;
var isBuffer = nativeIsBuffer || stubFalse_default;
var isBuffer_default = isBuffer;

// node_modules/lodash-es/_baseIsTypedArray.js
var argsTag2 = "[object Arguments]";
var arrayTag = "[object Array]";
var boolTag = "[object Boolean]";
var dateTag = "[object Date]";
var errorTag = "[object Error]";
var funcTag2 = "[object Function]";
var mapTag = "[object Map]";
var numberTag = "[object Number]";
var objectTag = "[object Object]";
var regexpTag = "[object RegExp]";
var setTag = "[object Set]";
var stringTag = "[object String]";
var weakMapTag = "[object WeakMap]";
var arrayBufferTag = "[object ArrayBuffer]";
var dataViewTag = "[object DataView]";
var float32Tag = "[object Float32Array]";
var float64Tag = "[object Float64Array]";
var int8Tag = "[object Int8Array]";
var int16Tag = "[object Int16Array]";
var int32Tag = "[object Int32Array]";
var uint8Tag = "[object Uint8Array]";
var uint8ClampedTag = "[object Uint8ClampedArray]";
var uint16Tag = "[object Uint16Array]";
var uint32Tag = "[object Uint32Array]";
var typedArrayTags = {};
typedArrayTags[float32Tag] = typedArrayTags[float64Tag] = typedArrayTags[int8Tag] = typedArrayTags[int16Tag] = typedArrayTags[int32Tag] = typedArrayTags[uint8Tag] = typedArrayTags[uint8ClampedTag] = typedArrayTags[uint16Tag] = typedArrayTags[uint32Tag] = true;
typedArrayTags[argsTag2] = typedArrayTags[arrayTag] = typedArrayTags[arrayBufferTag] = typedArrayTags[boolTag] = typedArrayTags[dataViewTag] = typedArrayTags[dateTag] = typedArrayTags[errorTag] = typedArrayTags[funcTag2] = typedArrayTags[mapTag] = typedArrayTags[numberTag] = typedArrayTags[objectTag] = typedArrayTags[regexpTag] = typedArrayTags[setTag] = typedArrayTags[stringTag] = typedArrayTags[weakMapTag] = false;
function baseIsTypedArray(value) {
  return isObjectLike_default(value) && isLength_default(value.length) && !!typedArrayTags[baseGetTag_default(value)];
}
var baseIsTypedArray_default = baseIsTypedArray;

// node_modules/lodash-es/_baseUnary.js
function baseUnary(func) {
  return function(value) {
    return func(value);
  };
}
var baseUnary_default = baseUnary;

// node_modules/lodash-es/_nodeUtil.js
var freeExports2 = typeof exports == "object" && exports && !exports.nodeType && exports;
var freeModule2 = freeExports2 && typeof module == "object" && module && !module.nodeType && module;
var moduleExports2 = freeModule2 && freeModule2.exports === freeExports2;
var freeProcess = moduleExports2 && freeGlobal_default.process;
var nodeUtil = function() {
  try {
    var types = freeModule2 && freeModule2.require && freeModule2.require("util").types;
    if (types) {
      return types;
    }
    return freeProcess && freeProcess.binding && freeProcess.binding("util");
  } catch (e) {
  }
}();
var nodeUtil_default = nodeUtil;

// node_modules/lodash-es/isTypedArray.js
var nodeIsTypedArray = nodeUtil_default && nodeUtil_default.isTypedArray;
var isTypedArray = nodeIsTypedArray ? baseUnary_default(nodeIsTypedArray) : baseIsTypedArray_default;
var isTypedArray_default = isTypedArray;

// node_modules/lodash-es/_arrayLikeKeys.js
var objectProto9 = Object.prototype;
var hasOwnProperty7 = objectProto9.hasOwnProperty;
function arrayLikeKeys(value, inherited) {
  var isArr = isArray_default(value), isArg = !isArr && isArguments_default(value), isBuff = !isArr && !isArg && isBuffer_default(value), isType2 = !isArr && !isArg && !isBuff && isTypedArray_default(value), skipIndexes = isArr || isArg || isBuff || isType2, result2 = skipIndexes ? baseTimes_default(value.length, String) : [], length = result2.length;
  for (var key in value) {
    if ((inherited || hasOwnProperty7.call(value, key)) && !(skipIndexes && // Safari 9 has enumerable `arguments.length` in strict mode.
    (key == "length" || // Node.js 0.10 has enumerable non-index properties on buffers.
    isBuff && (key == "offset" || key == "parent") || // PhantomJS 2 has enumerable non-index properties on typed arrays.
    isType2 && (key == "buffer" || key == "byteLength" || key == "byteOffset") || // Skip index properties.
    isIndex_default(key, length)))) {
      result2.push(key);
    }
  }
  return result2;
}
var arrayLikeKeys_default = arrayLikeKeys;

// node_modules/lodash-es/_overArg.js
function overArg(func, transform2) {
  return function(arg) {
    return func(transform2(arg));
  };
}
var overArg_default = overArg;

// node_modules/lodash-es/_nativeKeys.js
var nativeKeys = overArg_default(Object.keys, Object);
var nativeKeys_default = nativeKeys;

// node_modules/lodash-es/_baseKeys.js
var objectProto10 = Object.prototype;
var hasOwnProperty8 = objectProto10.hasOwnProperty;
function baseKeys(object) {
  if (!isPrototype_default(object)) {
    return nativeKeys_default(object);
  }
  var result2 = [];
  for (var key in Object(object)) {
    if (hasOwnProperty8.call(object, key) && key != "constructor") {
      result2.push(key);
    }
  }
  return result2;
}
var baseKeys_default = baseKeys;

// node_modules/lodash-es/keys.js
function keys(object) {
  return isArrayLike_default(object) ? arrayLikeKeys_default(object) : baseKeys_default(object);
}
var keys_default = keys;

// node_modules/lodash-es/assign.js
var objectProto11 = Object.prototype;
var hasOwnProperty9 = objectProto11.hasOwnProperty;
var assign = createAssigner_default(function(object, source) {
  if (isPrototype_default(source) || isArrayLike_default(source)) {
    copyObject_default(source, keys_default(source), object);
    return;
  }
  for (var key in source) {
    if (hasOwnProperty9.call(source, key)) {
      assignValue_default(object, key, source[key]);
    }
  }
});
var assign_default = assign;

// node_modules/lodash-es/_nativeKeysIn.js
function nativeKeysIn(object) {
  var result2 = [];
  if (object != null) {
    for (var key in Object(object)) {
      result2.push(key);
    }
  }
  return result2;
}
var nativeKeysIn_default = nativeKeysIn;

// node_modules/lodash-es/_baseKeysIn.js
var objectProto12 = Object.prototype;
var hasOwnProperty10 = objectProto12.hasOwnProperty;
function baseKeysIn(object) {
  if (!isObject_default(object)) {
    return nativeKeysIn_default(object);
  }
  var isProto = isPrototype_default(object), result2 = [];
  for (var key in object) {
    if (!(key == "constructor" && (isProto || !hasOwnProperty10.call(object, key)))) {
      result2.push(key);
    }
  }
  return result2;
}
var baseKeysIn_default = baseKeysIn;

// node_modules/lodash-es/keysIn.js
function keysIn(object) {
  return isArrayLike_default(object) ? arrayLikeKeys_default(object, true) : baseKeysIn_default(object);
}
var keysIn_default = keysIn;

// node_modules/lodash-es/assignIn.js
var assignIn = createAssigner_default(function(object, source) {
  copyObject_default(source, keysIn_default(source), object);
});
var assignIn_default = assignIn;

// node_modules/lodash-es/assignInWith.js
var assignInWith = createAssigner_default(function(object, source, srcIndex, customizer) {
  copyObject_default(source, keysIn_default(source), object, customizer);
});
var assignInWith_default = assignInWith;

// node_modules/lodash-es/assignWith.js
var assignWith = createAssigner_default(function(object, source, srcIndex, customizer) {
  copyObject_default(source, keys_default(source), object, customizer);
});
var assignWith_default = assignWith;

// node_modules/lodash-es/_isKey.js
var reIsDeepProp = /\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/;
var reIsPlainProp = /^\w*$/;
function isKey(value, object) {
  if (isArray_default(value)) {
    return false;
  }
  var type = typeof value;
  if (type == "number" || type == "symbol" || type == "boolean" || value == null || isSymbol_default(value)) {
    return true;
  }
  return reIsPlainProp.test(value) || !reIsDeepProp.test(value) || object != null && value in Object(object);
}
var isKey_default = isKey;

// node_modules/lodash-es/_nativeCreate.js
var nativeCreate = getNative_default(Object, "create");
var nativeCreate_default = nativeCreate;

// node_modules/lodash-es/_hashClear.js
function hashClear() {
  this.__data__ = nativeCreate_default ? nativeCreate_default(null) : {};
  this.size = 0;
}
var hashClear_default = hashClear;

// node_modules/lodash-es/_hashDelete.js
function hashDelete(key) {
  var result2 = this.has(key) && delete this.__data__[key];
  this.size -= result2 ? 1 : 0;
  return result2;
}
var hashDelete_default = hashDelete;

// node_modules/lodash-es/_hashGet.js
var HASH_UNDEFINED = "__lodash_hash_undefined__";
var objectProto13 = Object.prototype;
var hasOwnProperty11 = objectProto13.hasOwnProperty;
function hashGet(key) {
  var data = this.__data__;
  if (nativeCreate_default) {
    var result2 = data[key];
    return result2 === HASH_UNDEFINED ? void 0 : result2;
  }
  return hasOwnProperty11.call(data, key) ? data[key] : void 0;
}
var hashGet_default = hashGet;

// node_modules/lodash-es/_hashHas.js
var objectProto14 = Object.prototype;
var hasOwnProperty12 = objectProto14.hasOwnProperty;
function hashHas(key) {
  var data = this.__data__;
  return nativeCreate_default ? data[key] !== void 0 : hasOwnProperty12.call(data, key);
}
var hashHas_default = hashHas;

// node_modules/lodash-es/_hashSet.js
var HASH_UNDEFINED2 = "__lodash_hash_undefined__";
function hashSet(key, value) {
  var data = this.__data__;
  this.size += this.has(key) ? 0 : 1;
  data[key] = nativeCreate_default && value === void 0 ? HASH_UNDEFINED2 : value;
  return this;
}
var hashSet_default = hashSet;

// node_modules/lodash-es/_Hash.js
function Hash(entries) {
  var index = -1, length = entries == null ? 0 : entries.length;
  this.clear();
  while (++index < length) {
    var entry = entries[index];
    this.set(entry[0], entry[1]);
  }
}
Hash.prototype.clear = hashClear_default;
Hash.prototype["delete"] = hashDelete_default;
Hash.prototype.get = hashGet_default;
Hash.prototype.has = hashHas_default;
Hash.prototype.set = hashSet_default;
var Hash_default = Hash;

// node_modules/lodash-es/_listCacheClear.js
function listCacheClear() {
  this.__data__ = [];
  this.size = 0;
}
var listCacheClear_default = listCacheClear;

// node_modules/lodash-es/_assocIndexOf.js
function assocIndexOf(array, key) {
  var length = array.length;
  while (length--) {
    if (eq_default(array[length][0], key)) {
      return length;
    }
  }
  return -1;
}
var assocIndexOf_default = assocIndexOf;

// node_modules/lodash-es/_listCacheDelete.js
var arrayProto = Array.prototype;
var splice = arrayProto.splice;
function listCacheDelete(key) {
  var data = this.__data__, index = assocIndexOf_default(data, key);
  if (index < 0) {
    return false;
  }
  var lastIndex = data.length - 1;
  if (index == lastIndex) {
    data.pop();
  } else {
    splice.call(data, index, 1);
  }
  --this.size;
  return true;
}
var listCacheDelete_default = listCacheDelete;

// node_modules/lodash-es/_listCacheGet.js
function listCacheGet(key) {
  var data = this.__data__, index = assocIndexOf_default(data, key);
  return index < 0 ? void 0 : data[index][1];
}
var listCacheGet_default = listCacheGet;

// node_modules/lodash-es/_listCacheHas.js
function listCacheHas(key) {
  return assocIndexOf_default(this.__data__, key) > -1;
}
var listCacheHas_default = listCacheHas;

// node_modules/lodash-es/_listCacheSet.js
function listCacheSet(key, value) {
  var data = this.__data__, index = assocIndexOf_default(data, key);
  if (index < 0) {
    ++this.size;
    data.push([key, value]);
  } else {
    data[index][1] = value;
  }
  return this;
}
var listCacheSet_default = listCacheSet;

// node_modules/lodash-es/_ListCache.js
function ListCache(entries) {
  var index = -1, length = entries == null ? 0 : entries.length;
  this.clear();
  while (++index < length) {
    var entry = entries[index];
    this.set(entry[0], entry[1]);
  }
}
ListCache.prototype.clear = listCacheClear_default;
ListCache.prototype["delete"] = listCacheDelete_default;
ListCache.prototype.get = listCacheGet_default;
ListCache.prototype.has = listCacheHas_default;
ListCache.prototype.set = listCacheSet_default;
var ListCache_default = ListCache;

// node_modules/lodash-es/_Map.js
var Map2 = getNative_default(root_default, "Map");
var Map_default = Map2;

// node_modules/lodash-es/_mapCacheClear.js
function mapCacheClear() {
  this.size = 0;
  this.__data__ = {
    "hash": new Hash_default(),
    "map": new (Map_default || ListCache_default)(),
    "string": new Hash_default()
  };
}
var mapCacheClear_default = mapCacheClear;

// node_modules/lodash-es/_isKeyable.js
function isKeyable(value) {
  var type = typeof value;
  return type == "string" || type == "number" || type == "symbol" || type == "boolean" ? value !== "__proto__" : value === null;
}
var isKeyable_default = isKeyable;

// node_modules/lodash-es/_getMapData.js
function getMapData(map2, key) {
  var data = map2.__data__;
  return isKeyable_default(key) ? data[typeof key == "string" ? "string" : "hash"] : data.map;
}
var getMapData_default = getMapData;

// node_modules/lodash-es/_mapCacheDelete.js
function mapCacheDelete(key) {
  var result2 = getMapData_default(this, key)["delete"](key);
  this.size -= result2 ? 1 : 0;
  return result2;
}
var mapCacheDelete_default = mapCacheDelete;

// node_modules/lodash-es/_mapCacheGet.js
function mapCacheGet(key) {
  return getMapData_default(this, key).get(key);
}
var mapCacheGet_default = mapCacheGet;

// node_modules/lodash-es/_mapCacheHas.js
function mapCacheHas(key) {
  return getMapData_default(this, key).has(key);
}
var mapCacheHas_default = mapCacheHas;

// node_modules/lodash-es/_mapCacheSet.js
function mapCacheSet(key, value) {
  var data = getMapData_default(this, key), size2 = data.size;
  data.set(key, value);
  this.size += data.size == size2 ? 0 : 1;
  return this;
}
var mapCacheSet_default = mapCacheSet;

// node_modules/lodash-es/_MapCache.js
function MapCache(entries) {
  var index = -1, length = entries == null ? 0 : entries.length;
  this.clear();
  while (++index < length) {
    var entry = entries[index];
    this.set(entry[0], entry[1]);
  }
}
MapCache.prototype.clear = mapCacheClear_default;
MapCache.prototype["delete"] = mapCacheDelete_default;
MapCache.prototype.get = mapCacheGet_default;
MapCache.prototype.has = mapCacheHas_default;
MapCache.prototype.set = mapCacheSet_default;
var MapCache_default = MapCache;

// node_modules/lodash-es/memoize.js
var FUNC_ERROR_TEXT3 = "Expected a function";
function memoize(func, resolver) {
  if (typeof func != "function" || resolver != null && typeof resolver != "function") {
    throw new TypeError(FUNC_ERROR_TEXT3);
  }
  var memoized = function() {
    var args = arguments, key = resolver ? resolver.apply(this, args) : args[0], cache = memoized.cache;
    if (cache.has(key)) {
      return cache.get(key);
    }
    var result2 = func.apply(this, args);
    memoized.cache = cache.set(key, result2) || cache;
    return result2;
  };
  memoized.cache = new (memoize.Cache || MapCache_default)();
  return memoized;
}
memoize.Cache = MapCache_default;
var memoize_default = memoize;

// node_modules/lodash-es/_memoizeCapped.js
var MAX_MEMOIZE_SIZE = 500;
function memoizeCapped(func) {
  var result2 = memoize_default(func, function(key) {
    if (cache.size === MAX_MEMOIZE_SIZE) {
      cache.clear();
    }
    return key;
  });
  var cache = result2.cache;
  return result2;
}
var memoizeCapped_default = memoizeCapped;

// node_modules/lodash-es/_stringToPath.js
var rePropName = /[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g;
var reEscapeChar = /\\(\\)?/g;
var stringToPath = memoizeCapped_default(function(string) {
  var result2 = [];
  if (string.charCodeAt(0) === 46) {
    result2.push("");
  }
  string.replace(rePropName, function(match, number, quote, subString) {
    result2.push(quote ? subString.replace(reEscapeChar, "$1") : number || match);
  });
  return result2;
});
var stringToPath_default = stringToPath;

// node_modules/lodash-es/toString.js
function toString2(value) {
  return value == null ? "" : baseToString_default(value);
}
var toString_default = toString2;

// node_modules/lodash-es/_castPath.js
function castPath(value, object) {
  if (isArray_default(value)) {
    return value;
  }
  return isKey_default(value, object) ? [value] : stringToPath_default(toString_default(value));
}
var castPath_default = castPath;

// node_modules/lodash-es/_toKey.js
var INFINITY3 = 1 / 0;
function toKey(value) {
  if (typeof value == "string" || isSymbol_default(value)) {
    return value;
  }
  var result2 = value + "";
  return result2 == "0" && 1 / value == -INFINITY3 ? "-0" : result2;
}
var toKey_default = toKey;

// node_modules/lodash-es/_baseGet.js
function baseGet(object, path) {
  path = castPath_default(path, object);
  var index = 0, length = path.length;
  while (object != null && index < length) {
    object = object[toKey_default(path[index++])];
  }
  return index && index == length ? object : void 0;
}
var baseGet_default = baseGet;

// node_modules/lodash-es/get.js
function get(object, path, defaultValue) {
  var result2 = object == null ? void 0 : baseGet_default(object, path);
  return result2 === void 0 ? defaultValue : result2;
}
var get_default = get;

// node_modules/lodash-es/_baseAt.js
function baseAt(object, paths) {
  var index = -1, length = paths.length, result2 = Array(length), skip = object == null;
  while (++index < length) {
    result2[index] = skip ? void 0 : get_default(object, paths[index]);
  }
  return result2;
}
var baseAt_default = baseAt;

// node_modules/lodash-es/_arrayPush.js
function arrayPush(array, values2) {
  var index = -1, length = values2.length, offset = array.length;
  while (++index < length) {
    array[offset + index] = values2[index];
  }
  return array;
}
var arrayPush_default = arrayPush;

// node_modules/lodash-es/_isFlattenable.js
var spreadableSymbol = Symbol_default ? Symbol_default.isConcatSpreadable : void 0;
function isFlattenable(value) {
  return isArray_default(value) || isArguments_default(value) || !!(spreadableSymbol && value && value[spreadableSymbol]);
}
var isFlattenable_default = isFlattenable;

// node_modules/lodash-es/_baseFlatten.js
function baseFlatten(array, depth, predicate, isStrict, result2) {
  var index = -1, length = array.length;
  predicate || (predicate = isFlattenable_default);
  result2 || (result2 = []);
  while (++index < length) {
    var value = array[index];
    if (depth > 0 && predicate(value)) {
      if (depth > 1) {
        baseFlatten(value, depth - 1, predicate, isStrict, result2);
      } else {
        arrayPush_default(result2, value);
      }
    } else if (!isStrict) {
      result2[result2.length] = value;
    }
  }
  return result2;
}
var baseFlatten_default = baseFlatten;

// node_modules/lodash-es/flatten.js
function flatten(array) {
  var length = array == null ? 0 : array.length;
  return length ? baseFlatten_default(array, 1) : [];
}
var flatten_default = flatten;

// node_modules/lodash-es/_flatRest.js
function flatRest(func) {
  return setToString_default(overRest_default(func, void 0, flatten_default), func + "");
}
var flatRest_default = flatRest;

// node_modules/lodash-es/at.js
var at = flatRest_default(baseAt_default);
var at_default = at;

// node_modules/lodash-es/_getPrototype.js
var getPrototype = overArg_default(Object.getPrototypeOf, Object);
var getPrototype_default = getPrototype;

// node_modules/lodash-es/isPlainObject.js
var objectTag2 = "[object Object]";
var funcProto3 = Function.prototype;
var objectProto15 = Object.prototype;
var funcToString3 = funcProto3.toString;
var hasOwnProperty13 = objectProto15.hasOwnProperty;
var objectCtorString = funcToString3.call(Object);
function isPlainObject(value) {
  if (!isObjectLike_default(value) || baseGetTag_default(value) != objectTag2) {
    return false;
  }
  var proto = getPrototype_default(value);
  if (proto === null) {
    return true;
  }
  var Ctor = hasOwnProperty13.call(proto, "constructor") && proto.constructor;
  return typeof Ctor == "function" && Ctor instanceof Ctor && funcToString3.call(Ctor) == objectCtorString;
}
var isPlainObject_default = isPlainObject;

// node_modules/lodash-es/isError.js
var domExcTag = "[object DOMException]";
var errorTag2 = "[object Error]";
function isError(value) {
  if (!isObjectLike_default(value)) {
    return false;
  }
  var tag = baseGetTag_default(value);
  return tag == errorTag2 || tag == domExcTag || typeof value.message == "string" && typeof value.name == "string" && !isPlainObject_default(value);
}
var isError_default = isError;

// node_modules/lodash-es/attempt.js
var attempt = baseRest_default(function(func, args) {
  try {
    return apply_default(func, void 0, args);
  } catch (e) {
    return isError_default(e) ? e : new Error(e);
  }
});
var attempt_default = attempt;

// node_modules/lodash-es/before.js
var FUNC_ERROR_TEXT4 = "Expected a function";
function before(n, func) {
  var result2;
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT4);
  }
  n = toInteger_default(n);
  return function() {
    if (--n > 0) {
      result2 = func.apply(this, arguments);
    }
    if (n <= 1) {
      func = void 0;
    }
    return result2;
  };
}
var before_default = before;

// node_modules/lodash-es/bind.js
var WRAP_BIND_FLAG8 = 1;
var WRAP_PARTIAL_FLAG4 = 32;
var bind = baseRest_default(function(func, thisArg, partials) {
  var bitmask = WRAP_BIND_FLAG8;
  if (partials.length) {
    var holders = replaceHolders_default(partials, getHolder_default(bind));
    bitmask |= WRAP_PARTIAL_FLAG4;
  }
  return createWrap_default(func, bitmask, thisArg, partials, holders);
});
bind.placeholder = {};
var bind_default = bind;

// node_modules/lodash-es/bindAll.js
var bindAll = flatRest_default(function(object, methodNames) {
  arrayEach_default(methodNames, function(key) {
    key = toKey_default(key);
    baseAssignValue_default(object, key, bind_default(object[key], object));
  });
  return object;
});
var bindAll_default = bindAll;

// node_modules/lodash-es/bindKey.js
var WRAP_BIND_FLAG9 = 1;
var WRAP_BIND_KEY_FLAG6 = 2;
var WRAP_PARTIAL_FLAG5 = 32;
var bindKey = baseRest_default(function(object, key, partials) {
  var bitmask = WRAP_BIND_FLAG9 | WRAP_BIND_KEY_FLAG6;
  if (partials.length) {
    var holders = replaceHolders_default(partials, getHolder_default(bindKey));
    bitmask |= WRAP_PARTIAL_FLAG5;
  }
  return createWrap_default(key, bitmask, object, partials, holders);
});
bindKey.placeholder = {};
var bindKey_default = bindKey;

// node_modules/lodash-es/_baseSlice.js
function baseSlice(array, start, end) {
  var index = -1, length = array.length;
  if (start < 0) {
    start = -start > length ? 0 : length + start;
  }
  end = end > length ? length : end;
  if (end < 0) {
    end += length;
  }
  length = start > end ? 0 : end - start >>> 0;
  start >>>= 0;
  var result2 = Array(length);
  while (++index < length) {
    result2[index] = array[index + start];
  }
  return result2;
}
var baseSlice_default = baseSlice;

// node_modules/lodash-es/_castSlice.js
function castSlice(array, start, end) {
  var length = array.length;
  end = end === void 0 ? length : end;
  return !start && end >= length ? array : baseSlice_default(array, start, end);
}
var castSlice_default = castSlice;

// node_modules/lodash-es/_hasUnicode.js
var rsAstralRange = "\\ud800-\\udfff";
var rsComboMarksRange = "\\u0300-\\u036f";
var reComboHalfMarksRange = "\\ufe20-\\ufe2f";
var rsComboSymbolsRange = "\\u20d0-\\u20ff";
var rsComboRange = rsComboMarksRange + reComboHalfMarksRange + rsComboSymbolsRange;
var rsVarRange = "\\ufe0e\\ufe0f";
var rsZWJ = "\\u200d";
var reHasUnicode = RegExp("[" + rsZWJ + rsAstralRange + rsComboRange + rsVarRange + "]");
function hasUnicode(string) {
  return reHasUnicode.test(string);
}
var hasUnicode_default = hasUnicode;

// node_modules/lodash-es/_asciiToArray.js
function asciiToArray(string) {
  return string.split("");
}
var asciiToArray_default = asciiToArray;

// node_modules/lodash-es/_unicodeToArray.js
var rsAstralRange2 = "\\ud800-\\udfff";
var rsComboMarksRange2 = "\\u0300-\\u036f";
var reComboHalfMarksRange2 = "\\ufe20-\\ufe2f";
var rsComboSymbolsRange2 = "\\u20d0-\\u20ff";
var rsComboRange2 = rsComboMarksRange2 + reComboHalfMarksRange2 + rsComboSymbolsRange2;
var rsVarRange2 = "\\ufe0e\\ufe0f";
var rsAstral = "[" + rsAstralRange2 + "]";
var rsCombo = "[" + rsComboRange2 + "]";
var rsFitz = "\\ud83c[\\udffb-\\udfff]";
var rsModifier = "(?:" + rsCombo + "|" + rsFitz + ")";
var rsNonAstral = "[^" + rsAstralRange2 + "]";
var rsRegional = "(?:\\ud83c[\\udde6-\\uddff]){2}";
var rsSurrPair = "[\\ud800-\\udbff][\\udc00-\\udfff]";
var rsZWJ2 = "\\u200d";
var reOptMod = rsModifier + "?";
var rsOptVar = "[" + rsVarRange2 + "]?";
var rsOptJoin = "(?:" + rsZWJ2 + "(?:" + [rsNonAstral, rsRegional, rsSurrPair].join("|") + ")" + rsOptVar + reOptMod + ")*";
var rsSeq = rsOptVar + reOptMod + rsOptJoin;
var rsSymbol = "(?:" + [rsNonAstral + rsCombo + "?", rsCombo, rsRegional, rsSurrPair, rsAstral].join("|") + ")";
var reUnicode = RegExp(rsFitz + "(?=" + rsFitz + ")|" + rsSymbol + rsSeq, "g");
function unicodeToArray(string) {
  return string.match(reUnicode) || [];
}
var unicodeToArray_default = unicodeToArray;

// node_modules/lodash-es/_stringToArray.js
function stringToArray(string) {
  return hasUnicode_default(string) ? unicodeToArray_default(string) : asciiToArray_default(string);
}
var stringToArray_default = stringToArray;

// node_modules/lodash-es/_createCaseFirst.js
function createCaseFirst(methodName) {
  return function(string) {
    string = toString_default(string);
    var strSymbols = hasUnicode_default(string) ? stringToArray_default(string) : void 0;
    var chr = strSymbols ? strSymbols[0] : string.charAt(0);
    var trailing = strSymbols ? castSlice_default(strSymbols, 1).join("") : string.slice(1);
    return chr[methodName]() + trailing;
  };
}
var createCaseFirst_default = createCaseFirst;

// node_modules/lodash-es/upperFirst.js
var upperFirst = createCaseFirst_default("toUpperCase");
var upperFirst_default = upperFirst;

// node_modules/lodash-es/capitalize.js
function capitalize(string) {
  return upperFirst_default(toString_default(string).toLowerCase());
}
var capitalize_default = capitalize;

// node_modules/lodash-es/_arrayReduce.js
function arrayReduce(array, iteratee2, accumulator, initAccum) {
  var index = -1, length = array == null ? 0 : array.length;
  if (initAccum && length) {
    accumulator = array[++index];
  }
  while (++index < length) {
    accumulator = iteratee2(accumulator, array[index], index, array);
  }
  return accumulator;
}
var arrayReduce_default = arrayReduce;

// node_modules/lodash-es/_basePropertyOf.js
function basePropertyOf(object) {
  return function(key) {
    return object == null ? void 0 : object[key];
  };
}
var basePropertyOf_default = basePropertyOf;

// node_modules/lodash-es/_deburrLetter.js
var deburredLetters = {
  // Latin-1 Supplement block.
  "À": "A",
  "Á": "A",
  "Â": "A",
  "Ã": "A",
  "Ä": "A",
  "Å": "A",
  "à": "a",
  "á": "a",
  "â": "a",
  "ã": "a",
  "ä": "a",
  "å": "a",
  "Ç": "C",
  "ç": "c",
  "Ð": "D",
  "ð": "d",
  "È": "E",
  "É": "E",
  "Ê": "E",
  "Ë": "E",
  "è": "e",
  "é": "e",
  "ê": "e",
  "ë": "e",
  "Ì": "I",
  "Í": "I",
  "Î": "I",
  "Ï": "I",
  "ì": "i",
  "í": "i",
  "î": "i",
  "ï": "i",
  "Ñ": "N",
  "ñ": "n",
  "Ò": "O",
  "Ó": "O",
  "Ô": "O",
  "Õ": "O",
  "Ö": "O",
  "Ø": "O",
  "ò": "o",
  "ó": "o",
  "ô": "o",
  "õ": "o",
  "ö": "o",
  "ø": "o",
  "Ù": "U",
  "Ú": "U",
  "Û": "U",
  "Ü": "U",
  "ù": "u",
  "ú": "u",
  "û": "u",
  "ü": "u",
  "Ý": "Y",
  "ý": "y",
  "ÿ": "y",
  "Æ": "Ae",
  "æ": "ae",
  "Þ": "Th",
  "þ": "th",
  "ß": "ss",
  // Latin Extended-A block.
  "Ā": "A",
  "Ă": "A",
  "Ą": "A",
  "ā": "a",
  "ă": "a",
  "ą": "a",
  "Ć": "C",
  "Ĉ": "C",
  "Ċ": "C",
  "Č": "C",
  "ć": "c",
  "ĉ": "c",
  "ċ": "c",
  "č": "c",
  "Ď": "D",
  "Đ": "D",
  "ď": "d",
  "đ": "d",
  "Ē": "E",
  "Ĕ": "E",
  "Ė": "E",
  "Ę": "E",
  "Ě": "E",
  "ē": "e",
  "ĕ": "e",
  "ė": "e",
  "ę": "e",
  "ě": "e",
  "Ĝ": "G",
  "Ğ": "G",
  "Ġ": "G",
  "Ģ": "G",
  "ĝ": "g",
  "ğ": "g",
  "ġ": "g",
  "ģ": "g",
  "Ĥ": "H",
  "Ħ": "H",
  "ĥ": "h",
  "ħ": "h",
  "Ĩ": "I",
  "Ī": "I",
  "Ĭ": "I",
  "Į": "I",
  "İ": "I",
  "ĩ": "i",
  "ī": "i",
  "ĭ": "i",
  "į": "i",
  "ı": "i",
  "Ĵ": "J",
  "ĵ": "j",
  "Ķ": "K",
  "ķ": "k",
  "ĸ": "k",
  "Ĺ": "L",
  "Ļ": "L",
  "Ľ": "L",
  "Ŀ": "L",
  "Ł": "L",
  "ĺ": "l",
  "ļ": "l",
  "ľ": "l",
  "ŀ": "l",
  "ł": "l",
  "Ń": "N",
  "Ņ": "N",
  "Ň": "N",
  "Ŋ": "N",
  "ń": "n",
  "ņ": "n",
  "ň": "n",
  "ŋ": "n",
  "Ō": "O",
  "Ŏ": "O",
  "Ő": "O",
  "ō": "o",
  "ŏ": "o",
  "ő": "o",
  "Ŕ": "R",
  "Ŗ": "R",
  "Ř": "R",
  "ŕ": "r",
  "ŗ": "r",
  "ř": "r",
  "Ś": "S",
  "Ŝ": "S",
  "Ş": "S",
  "Š": "S",
  "ś": "s",
  "ŝ": "s",
  "ş": "s",
  "š": "s",
  "Ţ": "T",
  "Ť": "T",
  "Ŧ": "T",
  "ţ": "t",
  "ť": "t",
  "ŧ": "t",
  "Ũ": "U",
  "Ū": "U",
  "Ŭ": "U",
  "Ů": "U",
  "Ű": "U",
  "Ų": "U",
  "ũ": "u",
  "ū": "u",
  "ŭ": "u",
  "ů": "u",
  "ű": "u",
  "ų": "u",
  "Ŵ": "W",
  "ŵ": "w",
  "Ŷ": "Y",
  "ŷ": "y",
  "Ÿ": "Y",
  "Ź": "Z",
  "Ż": "Z",
  "Ž": "Z",
  "ź": "z",
  "ż": "z",
  "ž": "z",
  "Ĳ": "IJ",
  "ĳ": "ij",
  "Œ": "Oe",
  "œ": "oe",
  "ŉ": "'n",
  "ſ": "s"
};
var deburrLetter = basePropertyOf_default(deburredLetters);
var deburrLetter_default = deburrLetter;

// node_modules/lodash-es/deburr.js
var reLatin = /[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g;
var rsComboMarksRange3 = "\\u0300-\\u036f";
var reComboHalfMarksRange3 = "\\ufe20-\\ufe2f";
var rsComboSymbolsRange3 = "\\u20d0-\\u20ff";
var rsComboRange3 = rsComboMarksRange3 + reComboHalfMarksRange3 + rsComboSymbolsRange3;
var rsCombo2 = "[" + rsComboRange3 + "]";
var reComboMark = RegExp(rsCombo2, "g");
function deburr(string) {
  string = toString_default(string);
  return string && string.replace(reLatin, deburrLetter_default).replace(reComboMark, "");
}
var deburr_default = deburr;

// node_modules/lodash-es/_asciiWords.js
var reAsciiWord = /[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g;
function asciiWords(string) {
  return string.match(reAsciiWord) || [];
}
var asciiWords_default = asciiWords;

// node_modules/lodash-es/_hasUnicodeWord.js
var reHasUnicodeWord = /[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/;
function hasUnicodeWord(string) {
  return reHasUnicodeWord.test(string);
}
var hasUnicodeWord_default = hasUnicodeWord;

// node_modules/lodash-es/_unicodeWords.js
var rsAstralRange3 = "\\ud800-\\udfff";
var rsComboMarksRange4 = "\\u0300-\\u036f";
var reComboHalfMarksRange4 = "\\ufe20-\\ufe2f";
var rsComboSymbolsRange4 = "\\u20d0-\\u20ff";
var rsComboRange4 = rsComboMarksRange4 + reComboHalfMarksRange4 + rsComboSymbolsRange4;
var rsDingbatRange = "\\u2700-\\u27bf";
var rsLowerRange = "a-z\\xdf-\\xf6\\xf8-\\xff";
var rsMathOpRange = "\\xac\\xb1\\xd7\\xf7";
var rsNonCharRange = "\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf";
var rsPunctuationRange = "\\u2000-\\u206f";
var rsSpaceRange = " \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000";
var rsUpperRange = "A-Z\\xc0-\\xd6\\xd8-\\xde";
var rsVarRange3 = "\\ufe0e\\ufe0f";
var rsBreakRange = rsMathOpRange + rsNonCharRange + rsPunctuationRange + rsSpaceRange;
var rsApos = "['’]";
var rsBreak = "[" + rsBreakRange + "]";
var rsCombo3 = "[" + rsComboRange4 + "]";
var rsDigits = "\\d+";
var rsDingbat = "[" + rsDingbatRange + "]";
var rsLower = "[" + rsLowerRange + "]";
var rsMisc = "[^" + rsAstralRange3 + rsBreakRange + rsDigits + rsDingbatRange + rsLowerRange + rsUpperRange + "]";
var rsFitz2 = "\\ud83c[\\udffb-\\udfff]";
var rsModifier2 = "(?:" + rsCombo3 + "|" + rsFitz2 + ")";
var rsNonAstral2 = "[^" + rsAstralRange3 + "]";
var rsRegional2 = "(?:\\ud83c[\\udde6-\\uddff]){2}";
var rsSurrPair2 = "[\\ud800-\\udbff][\\udc00-\\udfff]";
var rsUpper = "[" + rsUpperRange + "]";
var rsZWJ3 = "\\u200d";
var rsMiscLower = "(?:" + rsLower + "|" + rsMisc + ")";
var rsMiscUpper = "(?:" + rsUpper + "|" + rsMisc + ")";
var rsOptContrLower = "(?:" + rsApos + "(?:d|ll|m|re|s|t|ve))?";
var rsOptContrUpper = "(?:" + rsApos + "(?:D|LL|M|RE|S|T|VE))?";
var reOptMod2 = rsModifier2 + "?";
var rsOptVar2 = "[" + rsVarRange3 + "]?";
var rsOptJoin2 = "(?:" + rsZWJ3 + "(?:" + [rsNonAstral2, rsRegional2, rsSurrPair2].join("|") + ")" + rsOptVar2 + reOptMod2 + ")*";
var rsOrdLower = "\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])";
var rsOrdUpper = "\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])";
var rsSeq2 = rsOptVar2 + reOptMod2 + rsOptJoin2;
var rsEmoji = "(?:" + [rsDingbat, rsRegional2, rsSurrPair2].join("|") + ")" + rsSeq2;
var reUnicodeWord = RegExp([
  rsUpper + "?" + rsLower + "+" + rsOptContrLower + "(?=" + [rsBreak, rsUpper, "$"].join("|") + ")",
  rsMiscUpper + "+" + rsOptContrUpper + "(?=" + [rsBreak, rsUpper + rsMiscLower, "$"].join("|") + ")",
  rsUpper + "?" + rsMiscLower + "+" + rsOptContrLower,
  rsUpper + "+" + rsOptContrUpper,
  rsOrdUpper,
  rsOrdLower,
  rsDigits,
  rsEmoji
].join("|"), "g");
function unicodeWords(string) {
  return string.match(reUnicodeWord) || [];
}
var unicodeWords_default = unicodeWords;

// node_modules/lodash-es/words.js
function words(string, pattern, guard) {
  string = toString_default(string);
  pattern = guard ? void 0 : pattern;
  if (pattern === void 0) {
    return hasUnicodeWord_default(string) ? unicodeWords_default(string) : asciiWords_default(string);
  }
  return string.match(pattern) || [];
}
var words_default = words;

// node_modules/lodash-es/_createCompounder.js
var rsApos2 = "['’]";
var reApos = RegExp(rsApos2, "g");
function createCompounder(callback) {
  return function(string) {
    return arrayReduce_default(words_default(deburr_default(string).replace(reApos, "")), callback, "");
  };
}
var createCompounder_default = createCompounder;

// node_modules/lodash-es/camelCase.js
var camelCase = createCompounder_default(function(result2, word, index) {
  word = word.toLowerCase();
  return result2 + (index ? capitalize_default(word) : word);
});
var camelCase_default = camelCase;

// node_modules/lodash-es/castArray.js
function castArray() {
  if (!arguments.length) {
    return [];
  }
  var value = arguments[0];
  return isArray_default(value) ? value : [value];
}
var castArray_default = castArray;

// node_modules/lodash-es/_createRound.js
var nativeIsFinite = root_default.isFinite;
var nativeMin3 = Math.min;
function createRound(methodName) {
  var func = Math[methodName];
  return function(number, precision) {
    number = toNumber_default(number);
    precision = precision == null ? 0 : nativeMin3(toInteger_default(precision), 292);
    if (precision && nativeIsFinite(number)) {
      var pair = (toString_default(number) + "e").split("e"), value = func(pair[0] + "e" + (+pair[1] + precision));
      pair = (toString_default(value) + "e").split("e");
      return +(pair[0] + "e" + (+pair[1] - precision));
    }
    return func(number);
  };
}
var createRound_default = createRound;

// node_modules/lodash-es/ceil.js
var ceil = createRound_default("ceil");
var ceil_default = ceil;

// node_modules/lodash-es/chain.js
function chain(value) {
  var result2 = wrapperLodash_default(value);
  result2.__chain__ = true;
  return result2;
}
var chain_default = chain;

// node_modules/lodash-es/chunk.js
var nativeCeil = Math.ceil;
var nativeMax5 = Math.max;
function chunk(array, size2, guard) {
  if (guard ? isIterateeCall_default(array, size2, guard) : size2 === void 0) {
    size2 = 1;
  } else {
    size2 = nativeMax5(toInteger_default(size2), 0);
  }
  var length = array == null ? 0 : array.length;
  if (!length || size2 < 1) {
    return [];
  }
  var index = 0, resIndex = 0, result2 = Array(nativeCeil(length / size2));
  while (index < length) {
    result2[resIndex++] = baseSlice_default(array, index, index += size2);
  }
  return result2;
}
var chunk_default = chunk;

// node_modules/lodash-es/_baseClamp.js
function baseClamp(number, lower, upper) {
  if (number === number) {
    if (upper !== void 0) {
      number = number <= upper ? number : upper;
    }
    if (lower !== void 0) {
      number = number >= lower ? number : lower;
    }
  }
  return number;
}
var baseClamp_default = baseClamp;

// node_modules/lodash-es/clamp.js
function clamp(number, lower, upper) {
  if (upper === void 0) {
    upper = lower;
    lower = void 0;
  }
  if (upper !== void 0) {
    upper = toNumber_default(upper);
    upper = upper === upper ? upper : 0;
  }
  if (lower !== void 0) {
    lower = toNumber_default(lower);
    lower = lower === lower ? lower : 0;
  }
  return baseClamp_default(toNumber_default(number), lower, upper);
}
var clamp_default = clamp;

// node_modules/lodash-es/_stackClear.js
function stackClear() {
  this.__data__ = new ListCache_default();
  this.size = 0;
}
var stackClear_default = stackClear;

// node_modules/lodash-es/_stackDelete.js
function stackDelete(key) {
  var data = this.__data__, result2 = data["delete"](key);
  this.size = data.size;
  return result2;
}
var stackDelete_default = stackDelete;

// node_modules/lodash-es/_stackGet.js
function stackGet(key) {
  return this.__data__.get(key);
}
var stackGet_default = stackGet;

// node_modules/lodash-es/_stackHas.js
function stackHas(key) {
  return this.__data__.has(key);
}
var stackHas_default = stackHas;

// node_modules/lodash-es/_stackSet.js
var LARGE_ARRAY_SIZE = 200;
function stackSet(key, value) {
  var data = this.__data__;
  if (data instanceof ListCache_default) {
    var pairs = data.__data__;
    if (!Map_default || pairs.length < LARGE_ARRAY_SIZE - 1) {
      pairs.push([key, value]);
      this.size = ++data.size;
      return this;
    }
    data = this.__data__ = new MapCache_default(pairs);
  }
  data.set(key, value);
  this.size = data.size;
  return this;
}
var stackSet_default = stackSet;

// node_modules/lodash-es/_Stack.js
function Stack(entries) {
  var data = this.__data__ = new ListCache_default(entries);
  this.size = data.size;
}
Stack.prototype.clear = stackClear_default;
Stack.prototype["delete"] = stackDelete_default;
Stack.prototype.get = stackGet_default;
Stack.prototype.has = stackHas_default;
Stack.prototype.set = stackSet_default;
var Stack_default = Stack;

// node_modules/lodash-es/_baseAssign.js
function baseAssign(object, source) {
  return object && copyObject_default(source, keys_default(source), object);
}
var baseAssign_default = baseAssign;

// node_modules/lodash-es/_baseAssignIn.js
function baseAssignIn(object, source) {
  return object && copyObject_default(source, keysIn_default(source), object);
}
var baseAssignIn_default = baseAssignIn;

// node_modules/lodash-es/_cloneBuffer.js
var freeExports3 = typeof exports == "object" && exports && !exports.nodeType && exports;
var freeModule3 = freeExports3 && typeof module == "object" && module && !module.nodeType && module;
var moduleExports3 = freeModule3 && freeModule3.exports === freeExports3;
var Buffer2 = moduleExports3 ? root_default.Buffer : void 0;
var allocUnsafe = Buffer2 ? Buffer2.allocUnsafe : void 0;
function cloneBuffer(buffer, isDeep) {
  if (isDeep) {
    return buffer.slice();
  }
  var length = buffer.length, result2 = allocUnsafe ? allocUnsafe(length) : new buffer.constructor(length);
  buffer.copy(result2);
  return result2;
}
var cloneBuffer_default = cloneBuffer;

// node_modules/lodash-es/_arrayFilter.js
function arrayFilter(array, predicate) {
  var index = -1, length = array == null ? 0 : array.length, resIndex = 0, result2 = [];
  while (++index < length) {
    var value = array[index];
    if (predicate(value, index, array)) {
      result2[resIndex++] = value;
    }
  }
  return result2;
}
var arrayFilter_default = arrayFilter;

// node_modules/lodash-es/stubArray.js
function stubArray() {
  return [];
}
var stubArray_default = stubArray;

// node_modules/lodash-es/_getSymbols.js
var objectProto16 = Object.prototype;
var propertyIsEnumerable2 = objectProto16.propertyIsEnumerable;
var nativeGetSymbols = Object.getOwnPropertySymbols;
var getSymbols = !nativeGetSymbols ? stubArray_default : function(object) {
  if (object == null) {
    return [];
  }
  object = Object(object);
  return arrayFilter_default(nativeGetSymbols(object), function(symbol) {
    return propertyIsEnumerable2.call(object, symbol);
  });
};
var getSymbols_default = getSymbols;

// node_modules/lodash-es/_copySymbols.js
function copySymbols(source, object) {
  return copyObject_default(source, getSymbols_default(source), object);
}
var copySymbols_default = copySymbols;

// node_modules/lodash-es/_getSymbolsIn.js
var nativeGetSymbols2 = Object.getOwnPropertySymbols;
var getSymbolsIn = !nativeGetSymbols2 ? stubArray_default : function(object) {
  var result2 = [];
  while (object) {
    arrayPush_default(result2, getSymbols_default(object));
    object = getPrototype_default(object);
  }
  return result2;
};
var getSymbolsIn_default = getSymbolsIn;

// node_modules/lodash-es/_copySymbolsIn.js
function copySymbolsIn(source, object) {
  return copyObject_default(source, getSymbolsIn_default(source), object);
}
var copySymbolsIn_default = copySymbolsIn;

// node_modules/lodash-es/_baseGetAllKeys.js
function baseGetAllKeys(object, keysFunc, symbolsFunc) {
  var result2 = keysFunc(object);
  return isArray_default(object) ? result2 : arrayPush_default(result2, symbolsFunc(object));
}
var baseGetAllKeys_default = baseGetAllKeys;

// node_modules/lodash-es/_getAllKeys.js
function getAllKeys(object) {
  return baseGetAllKeys_default(object, keys_default, getSymbols_default);
}
var getAllKeys_default = getAllKeys;

// node_modules/lodash-es/_getAllKeysIn.js
function getAllKeysIn(object) {
  return baseGetAllKeys_default(object, keysIn_default, getSymbolsIn_default);
}
var getAllKeysIn_default = getAllKeysIn;

// node_modules/lodash-es/_DataView.js
var DataView = getNative_default(root_default, "DataView");
var DataView_default = DataView;

// node_modules/lodash-es/_Promise.js
var Promise2 = getNative_default(root_default, "Promise");
var Promise_default = Promise2;

// node_modules/lodash-es/_Set.js
var Set2 = getNative_default(root_default, "Set");
var Set_default = Set2;

// node_modules/lodash-es/_getTag.js
var mapTag2 = "[object Map]";
var objectTag3 = "[object Object]";
var promiseTag = "[object Promise]";
var setTag2 = "[object Set]";
var weakMapTag2 = "[object WeakMap]";
var dataViewTag2 = "[object DataView]";
var dataViewCtorString = toSource_default(DataView_default);
var mapCtorString = toSource_default(Map_default);
var promiseCtorString = toSource_default(Promise_default);
var setCtorString = toSource_default(Set_default);
var weakMapCtorString = toSource_default(WeakMap_default);
var getTag = baseGetTag_default;
if (DataView_default && getTag(new DataView_default(new ArrayBuffer(1))) != dataViewTag2 || Map_default && getTag(new Map_default()) != mapTag2 || Promise_default && getTag(Promise_default.resolve()) != promiseTag || Set_default && getTag(new Set_default()) != setTag2 || WeakMap_default && getTag(new WeakMap_default()) != weakMapTag2) {
  getTag = function(value) {
    var result2 = baseGetTag_default(value), Ctor = result2 == objectTag3 ? value.constructor : void 0, ctorString = Ctor ? toSource_default(Ctor) : "";
    if (ctorString) {
      switch (ctorString) {
        case dataViewCtorString:
          return dataViewTag2;
        case mapCtorString:
          return mapTag2;
        case promiseCtorString:
          return promiseTag;
        case setCtorString:
          return setTag2;
        case weakMapCtorString:
          return weakMapTag2;
      }
    }
    return result2;
  };
}
var getTag_default = getTag;

// node_modules/lodash-es/_initCloneArray.js
var objectProto17 = Object.prototype;
var hasOwnProperty14 = objectProto17.hasOwnProperty;
function initCloneArray(array) {
  var length = array.length, result2 = new array.constructor(length);
  if (length && typeof array[0] == "string" && hasOwnProperty14.call(array, "index")) {
    result2.index = array.index;
    result2.input = array.input;
  }
  return result2;
}
var initCloneArray_default = initCloneArray;

// node_modules/lodash-es/_Uint8Array.js
var Uint8Array = root_default.Uint8Array;
var Uint8Array_default = Uint8Array;

// node_modules/lodash-es/_cloneArrayBuffer.js
function cloneArrayBuffer(arrayBuffer) {
  var result2 = new arrayBuffer.constructor(arrayBuffer.byteLength);
  new Uint8Array_default(result2).set(new Uint8Array_default(arrayBuffer));
  return result2;
}
var cloneArrayBuffer_default = cloneArrayBuffer;

// node_modules/lodash-es/_cloneDataView.js
function cloneDataView(dataView, isDeep) {
  var buffer = isDeep ? cloneArrayBuffer_default(dataView.buffer) : dataView.buffer;
  return new dataView.constructor(buffer, dataView.byteOffset, dataView.byteLength);
}
var cloneDataView_default = cloneDataView;

// node_modules/lodash-es/_cloneRegExp.js
var reFlags = /\w*$/;
function cloneRegExp(regexp) {
  var result2 = new regexp.constructor(regexp.source, reFlags.exec(regexp));
  result2.lastIndex = regexp.lastIndex;
  return result2;
}
var cloneRegExp_default = cloneRegExp;

// node_modules/lodash-es/_cloneSymbol.js
var symbolProto2 = Symbol_default ? Symbol_default.prototype : void 0;
var symbolValueOf = symbolProto2 ? symbolProto2.valueOf : void 0;
function cloneSymbol(symbol) {
  return symbolValueOf ? Object(symbolValueOf.call(symbol)) : {};
}
var cloneSymbol_default = cloneSymbol;

// node_modules/lodash-es/_cloneTypedArray.js
function cloneTypedArray(typedArray, isDeep) {
  var buffer = isDeep ? cloneArrayBuffer_default(typedArray.buffer) : typedArray.buffer;
  return new typedArray.constructor(buffer, typedArray.byteOffset, typedArray.length);
}
var cloneTypedArray_default = cloneTypedArray;

// node_modules/lodash-es/_initCloneByTag.js
var boolTag2 = "[object Boolean]";
var dateTag2 = "[object Date]";
var mapTag3 = "[object Map]";
var numberTag2 = "[object Number]";
var regexpTag2 = "[object RegExp]";
var setTag3 = "[object Set]";
var stringTag2 = "[object String]";
var symbolTag2 = "[object Symbol]";
var arrayBufferTag2 = "[object ArrayBuffer]";
var dataViewTag3 = "[object DataView]";
var float32Tag2 = "[object Float32Array]";
var float64Tag2 = "[object Float64Array]";
var int8Tag2 = "[object Int8Array]";
var int16Tag2 = "[object Int16Array]";
var int32Tag2 = "[object Int32Array]";
var uint8Tag2 = "[object Uint8Array]";
var uint8ClampedTag2 = "[object Uint8ClampedArray]";
var uint16Tag2 = "[object Uint16Array]";
var uint32Tag2 = "[object Uint32Array]";
function initCloneByTag(object, tag, isDeep) {
  var Ctor = object.constructor;
  switch (tag) {
    case arrayBufferTag2:
      return cloneArrayBuffer_default(object);
    case boolTag2:
    case dateTag2:
      return new Ctor(+object);
    case dataViewTag3:
      return cloneDataView_default(object, isDeep);
    case float32Tag2:
    case float64Tag2:
    case int8Tag2:
    case int16Tag2:
    case int32Tag2:
    case uint8Tag2:
    case uint8ClampedTag2:
    case uint16Tag2:
    case uint32Tag2:
      return cloneTypedArray_default(object, isDeep);
    case mapTag3:
      return new Ctor();
    case numberTag2:
    case stringTag2:
      return new Ctor(object);
    case regexpTag2:
      return cloneRegExp_default(object);
    case setTag3:
      return new Ctor();
    case symbolTag2:
      return cloneSymbol_default(object);
  }
}
var initCloneByTag_default = initCloneByTag;

// node_modules/lodash-es/_initCloneObject.js
function initCloneObject(object) {
  return typeof object.constructor == "function" && !isPrototype_default(object) ? baseCreate_default(getPrototype_default(object)) : {};
}
var initCloneObject_default = initCloneObject;

// node_modules/lodash-es/_baseIsMap.js
var mapTag4 = "[object Map]";
function baseIsMap(value) {
  return isObjectLike_default(value) && getTag_default(value) == mapTag4;
}
var baseIsMap_default = baseIsMap;

// node_modules/lodash-es/isMap.js
var nodeIsMap = nodeUtil_default && nodeUtil_default.isMap;
var isMap = nodeIsMap ? baseUnary_default(nodeIsMap) : baseIsMap_default;
var isMap_default = isMap;

// node_modules/lodash-es/_baseIsSet.js
var setTag4 = "[object Set]";
function baseIsSet(value) {
  return isObjectLike_default(value) && getTag_default(value) == setTag4;
}
var baseIsSet_default = baseIsSet;

// node_modules/lodash-es/isSet.js
var nodeIsSet = nodeUtil_default && nodeUtil_default.isSet;
var isSet = nodeIsSet ? baseUnary_default(nodeIsSet) : baseIsSet_default;
var isSet_default = isSet;

// node_modules/lodash-es/_baseClone.js
var CLONE_DEEP_FLAG = 1;
var CLONE_FLAT_FLAG = 2;
var CLONE_SYMBOLS_FLAG = 4;
var argsTag3 = "[object Arguments]";
var arrayTag2 = "[object Array]";
var boolTag3 = "[object Boolean]";
var dateTag3 = "[object Date]";
var errorTag3 = "[object Error]";
var funcTag3 = "[object Function]";
var genTag2 = "[object GeneratorFunction]";
var mapTag5 = "[object Map]";
var numberTag3 = "[object Number]";
var objectTag4 = "[object Object]";
var regexpTag3 = "[object RegExp]";
var setTag5 = "[object Set]";
var stringTag3 = "[object String]";
var symbolTag3 = "[object Symbol]";
var weakMapTag3 = "[object WeakMap]";
var arrayBufferTag3 = "[object ArrayBuffer]";
var dataViewTag4 = "[object DataView]";
var float32Tag3 = "[object Float32Array]";
var float64Tag3 = "[object Float64Array]";
var int8Tag3 = "[object Int8Array]";
var int16Tag3 = "[object Int16Array]";
var int32Tag3 = "[object Int32Array]";
var uint8Tag3 = "[object Uint8Array]";
var uint8ClampedTag3 = "[object Uint8ClampedArray]";
var uint16Tag3 = "[object Uint16Array]";
var uint32Tag3 = "[object Uint32Array]";
var cloneableTags = {};
cloneableTags[argsTag3] = cloneableTags[arrayTag2] = cloneableTags[arrayBufferTag3] = cloneableTags[dataViewTag4] = cloneableTags[boolTag3] = cloneableTags[dateTag3] = cloneableTags[float32Tag3] = cloneableTags[float64Tag3] = cloneableTags[int8Tag3] = cloneableTags[int16Tag3] = cloneableTags[int32Tag3] = cloneableTags[mapTag5] = cloneableTags[numberTag3] = cloneableTags[objectTag4] = cloneableTags[regexpTag3] = cloneableTags[setTag5] = cloneableTags[stringTag3] = cloneableTags[symbolTag3] = cloneableTags[uint8Tag3] = cloneableTags[uint8ClampedTag3] = cloneableTags[uint16Tag3] = cloneableTags[uint32Tag3] = true;
cloneableTags[errorTag3] = cloneableTags[funcTag3] = cloneableTags[weakMapTag3] = false;
function baseClone(value, bitmask, customizer, key, object, stack) {
  var result2, isDeep = bitmask & CLONE_DEEP_FLAG, isFlat = bitmask & CLONE_FLAT_FLAG, isFull = bitmask & CLONE_SYMBOLS_FLAG;
  if (customizer) {
    result2 = object ? customizer(value, key, object, stack) : customizer(value);
  }
  if (result2 !== void 0) {
    return result2;
  }
  if (!isObject_default(value)) {
    return value;
  }
  var isArr = isArray_default(value);
  if (isArr) {
    result2 = initCloneArray_default(value);
    if (!isDeep) {
      return copyArray_default(value, result2);
    }
  } else {
    var tag = getTag_default(value), isFunc = tag == funcTag3 || tag == genTag2;
    if (isBuffer_default(value)) {
      return cloneBuffer_default(value, isDeep);
    }
    if (tag == objectTag4 || tag == argsTag3 || isFunc && !object) {
      result2 = isFlat || isFunc ? {} : initCloneObject_default(value);
      if (!isDeep) {
        return isFlat ? copySymbolsIn_default(value, baseAssignIn_default(result2, value)) : copySymbols_default(value, baseAssign_default(result2, value));
      }
    } else {
      if (!cloneableTags[tag]) {
        return object ? value : {};
      }
      result2 = initCloneByTag_default(value, tag, isDeep);
    }
  }
  stack || (stack = new Stack_default());
  var stacked = stack.get(value);
  if (stacked) {
    return stacked;
  }
  stack.set(value, result2);
  if (isSet_default(value)) {
    value.forEach(function(subValue) {
      result2.add(baseClone(subValue, bitmask, customizer, subValue, value, stack));
    });
  } else if (isMap_default(value)) {
    value.forEach(function(subValue, key2) {
      result2.set(key2, baseClone(subValue, bitmask, customizer, key2, value, stack));
    });
  }
  var keysFunc = isFull ? isFlat ? getAllKeysIn_default : getAllKeys_default : isFlat ? keysIn_default : keys_default;
  var props = isArr ? void 0 : keysFunc(value);
  arrayEach_default(props || value, function(subValue, key2) {
    if (props) {
      key2 = subValue;
      subValue = value[key2];
    }
    assignValue_default(result2, key2, baseClone(subValue, bitmask, customizer, key2, value, stack));
  });
  return result2;
}
var baseClone_default = baseClone;

// node_modules/lodash-es/clone.js
var CLONE_SYMBOLS_FLAG2 = 4;
function clone(value) {
  return baseClone_default(value, CLONE_SYMBOLS_FLAG2);
}
var clone_default = clone;

// node_modules/lodash-es/cloneDeep.js
var CLONE_DEEP_FLAG2 = 1;
var CLONE_SYMBOLS_FLAG3 = 4;
function cloneDeep(value) {
  return baseClone_default(value, CLONE_DEEP_FLAG2 | CLONE_SYMBOLS_FLAG3);
}
var cloneDeep_default = cloneDeep;

// node_modules/lodash-es/cloneDeepWith.js
var CLONE_DEEP_FLAG3 = 1;
var CLONE_SYMBOLS_FLAG4 = 4;
function cloneDeepWith(value, customizer) {
  customizer = typeof customizer == "function" ? customizer : void 0;
  return baseClone_default(value, CLONE_DEEP_FLAG3 | CLONE_SYMBOLS_FLAG4, customizer);
}
var cloneDeepWith_default = cloneDeepWith;

// node_modules/lodash-es/cloneWith.js
var CLONE_SYMBOLS_FLAG5 = 4;
function cloneWith(value, customizer) {
  customizer = typeof customizer == "function" ? customizer : void 0;
  return baseClone_default(value, CLONE_SYMBOLS_FLAG5, customizer);
}
var cloneWith_default = cloneWith;

// node_modules/lodash-es/commit.js
function wrapperCommit() {
  return new LodashWrapper_default(this.value(), this.__chain__);
}
var commit_default = wrapperCommit;

// node_modules/lodash-es/compact.js
function compact(array) {
  var index = -1, length = array == null ? 0 : array.length, resIndex = 0, result2 = [];
  while (++index < length) {
    var value = array[index];
    if (value) {
      result2[resIndex++] = value;
    }
  }
  return result2;
}
var compact_default = compact;

// node_modules/lodash-es/concat.js
function concat() {
  var length = arguments.length;
  if (!length) {
    return [];
  }
  var args = Array(length - 1), array = arguments[0], index = length;
  while (index--) {
    args[index - 1] = arguments[index];
  }
  return arrayPush_default(isArray_default(array) ? copyArray_default(array) : [array], baseFlatten_default(args, 1));
}
var concat_default = concat;

// node_modules/lodash-es/_setCacheAdd.js
var HASH_UNDEFINED3 = "__lodash_hash_undefined__";
function setCacheAdd(value) {
  this.__data__.set(value, HASH_UNDEFINED3);
  return this;
}
var setCacheAdd_default = setCacheAdd;

// node_modules/lodash-es/_setCacheHas.js
function setCacheHas(value) {
  return this.__data__.has(value);
}
var setCacheHas_default = setCacheHas;

// node_modules/lodash-es/_SetCache.js
function SetCache(values2) {
  var index = -1, length = values2 == null ? 0 : values2.length;
  this.__data__ = new MapCache_default();
  while (++index < length) {
    this.add(values2[index]);
  }
}
SetCache.prototype.add = SetCache.prototype.push = setCacheAdd_default;
SetCache.prototype.has = setCacheHas_default;
var SetCache_default = SetCache;

// node_modules/lodash-es/_arraySome.js
function arraySome(array, predicate) {
  var index = -1, length = array == null ? 0 : array.length;
  while (++index < length) {
    if (predicate(array[index], index, array)) {
      return true;
    }
  }
  return false;
}
var arraySome_default = arraySome;

// node_modules/lodash-es/_cacheHas.js
function cacheHas(cache, key) {
  return cache.has(key);
}
var cacheHas_default = cacheHas;

// node_modules/lodash-es/_equalArrays.js
var COMPARE_PARTIAL_FLAG = 1;
var COMPARE_UNORDERED_FLAG = 2;
function equalArrays(array, other, bitmask, customizer, equalFunc, stack) {
  var isPartial = bitmask & COMPARE_PARTIAL_FLAG, arrLength = array.length, othLength = other.length;
  if (arrLength != othLength && !(isPartial && othLength > arrLength)) {
    return false;
  }
  var arrStacked = stack.get(array);
  var othStacked = stack.get(other);
  if (arrStacked && othStacked) {
    return arrStacked == other && othStacked == array;
  }
  var index = -1, result2 = true, seen = bitmask & COMPARE_UNORDERED_FLAG ? new SetCache_default() : void 0;
  stack.set(array, other);
  stack.set(other, array);
  while (++index < arrLength) {
    var arrValue = array[index], othValue = other[index];
    if (customizer) {
      var compared = isPartial ? customizer(othValue, arrValue, index, other, array, stack) : customizer(arrValue, othValue, index, array, other, stack);
    }
    if (compared !== void 0) {
      if (compared) {
        continue;
      }
      result2 = false;
      break;
    }
    if (seen) {
      if (!arraySome_default(other, function(othValue2, othIndex) {
        if (!cacheHas_default(seen, othIndex) && (arrValue === othValue2 || equalFunc(arrValue, othValue2, bitmask, customizer, stack))) {
          return seen.push(othIndex);
        }
      })) {
        result2 = false;
        break;
      }
    } else if (!(arrValue === othValue || equalFunc(arrValue, othValue, bitmask, customizer, stack))) {
      result2 = false;
      break;
    }
  }
  stack["delete"](array);
  stack["delete"](other);
  return result2;
}
var equalArrays_default = equalArrays;

// node_modules/lodash-es/_mapToArray.js
function mapToArray(map2) {
  var index = -1, result2 = Array(map2.size);
  map2.forEach(function(value, key) {
    result2[++index] = [key, value];
  });
  return result2;
}
var mapToArray_default = mapToArray;

// node_modules/lodash-es/_setToArray.js
function setToArray(set2) {
  var index = -1, result2 = Array(set2.size);
  set2.forEach(function(value) {
    result2[++index] = value;
  });
  return result2;
}
var setToArray_default = setToArray;

// node_modules/lodash-es/_equalByTag.js
var COMPARE_PARTIAL_FLAG2 = 1;
var COMPARE_UNORDERED_FLAG2 = 2;
var boolTag4 = "[object Boolean]";
var dateTag4 = "[object Date]";
var errorTag4 = "[object Error]";
var mapTag6 = "[object Map]";
var numberTag4 = "[object Number]";
var regexpTag4 = "[object RegExp]";
var setTag6 = "[object Set]";
var stringTag4 = "[object String]";
var symbolTag4 = "[object Symbol]";
var arrayBufferTag4 = "[object ArrayBuffer]";
var dataViewTag5 = "[object DataView]";
var symbolProto3 = Symbol_default ? Symbol_default.prototype : void 0;
var symbolValueOf2 = symbolProto3 ? symbolProto3.valueOf : void 0;
function equalByTag(object, other, tag, bitmask, customizer, equalFunc, stack) {
  switch (tag) {
    case dataViewTag5:
      if (object.byteLength != other.byteLength || object.byteOffset != other.byteOffset) {
        return false;
      }
      object = object.buffer;
      other = other.buffer;
    case arrayBufferTag4:
      if (object.byteLength != other.byteLength || !equalFunc(new Uint8Array_default(object), new Uint8Array_default(other))) {
        return false;
      }
      return true;
    case boolTag4:
    case dateTag4:
    case numberTag4:
      return eq_default(+object, +other);
    case errorTag4:
      return object.name == other.name && object.message == other.message;
    case regexpTag4:
    case stringTag4:
      return object == other + "";
    case mapTag6:
      var convert = mapToArray_default;
    case setTag6:
      var isPartial = bitmask & COMPARE_PARTIAL_FLAG2;
      convert || (convert = setToArray_default);
      if (object.size != other.size && !isPartial) {
        return false;
      }
      var stacked = stack.get(object);
      if (stacked) {
        return stacked == other;
      }
      bitmask |= COMPARE_UNORDERED_FLAG2;
      stack.set(object, other);
      var result2 = equalArrays_default(convert(object), convert(other), bitmask, customizer, equalFunc, stack);
      stack["delete"](object);
      return result2;
    case symbolTag4:
      if (symbolValueOf2) {
        return symbolValueOf2.call(object) == symbolValueOf2.call(other);
      }
  }
  return false;
}
var equalByTag_default = equalByTag;

// node_modules/lodash-es/_equalObjects.js
var COMPARE_PARTIAL_FLAG3 = 1;
var objectProto18 = Object.prototype;
var hasOwnProperty15 = objectProto18.hasOwnProperty;
function equalObjects(object, other, bitmask, customizer, equalFunc, stack) {
  var isPartial = bitmask & COMPARE_PARTIAL_FLAG3, objProps = getAllKeys_default(object), objLength = objProps.length, othProps = getAllKeys_default(other), othLength = othProps.length;
  if (objLength != othLength && !isPartial) {
    return false;
  }
  var index = objLength;
  while (index--) {
    var key = objProps[index];
    if (!(isPartial ? key in other : hasOwnProperty15.call(other, key))) {
      return false;
    }
  }
  var objStacked = stack.get(object);
  var othStacked = stack.get(other);
  if (objStacked && othStacked) {
    return objStacked == other && othStacked == object;
  }
  var result2 = true;
  stack.set(object, other);
  stack.set(other, object);
  var skipCtor = isPartial;
  while (++index < objLength) {
    key = objProps[index];
    var objValue = object[key], othValue = other[key];
    if (customizer) {
      var compared = isPartial ? customizer(othValue, objValue, key, other, object, stack) : customizer(objValue, othValue, key, object, other, stack);
    }
    if (!(compared === void 0 ? objValue === othValue || equalFunc(objValue, othValue, bitmask, customizer, stack) : compared)) {
      result2 = false;
      break;
    }
    skipCtor || (skipCtor = key == "constructor");
  }
  if (result2 && !skipCtor) {
    var objCtor = object.constructor, othCtor = other.constructor;
    if (objCtor != othCtor && ("constructor" in object && "constructor" in other) && !(typeof objCtor == "function" && objCtor instanceof objCtor && typeof othCtor == "function" && othCtor instanceof othCtor)) {
      result2 = false;
    }
  }
  stack["delete"](object);
  stack["delete"](other);
  return result2;
}
var equalObjects_default = equalObjects;

// node_modules/lodash-es/_baseIsEqualDeep.js
var COMPARE_PARTIAL_FLAG4 = 1;
var argsTag4 = "[object Arguments]";
var arrayTag3 = "[object Array]";
var objectTag5 = "[object Object]";
var objectProto19 = Object.prototype;
var hasOwnProperty16 = objectProto19.hasOwnProperty;
function baseIsEqualDeep(object, other, bitmask, customizer, equalFunc, stack) {
  var objIsArr = isArray_default(object), othIsArr = isArray_default(other), objTag = objIsArr ? arrayTag3 : getTag_default(object), othTag = othIsArr ? arrayTag3 : getTag_default(other);
  objTag = objTag == argsTag4 ? objectTag5 : objTag;
  othTag = othTag == argsTag4 ? objectTag5 : othTag;
  var objIsObj = objTag == objectTag5, othIsObj = othTag == objectTag5, isSameTag = objTag == othTag;
  if (isSameTag && isBuffer_default(object)) {
    if (!isBuffer_default(other)) {
      return false;
    }
    objIsArr = true;
    objIsObj = false;
  }
  if (isSameTag && !objIsObj) {
    stack || (stack = new Stack_default());
    return objIsArr || isTypedArray_default(object) ? equalArrays_default(object, other, bitmask, customizer, equalFunc, stack) : equalByTag_default(object, other, objTag, bitmask, customizer, equalFunc, stack);
  }
  if (!(bitmask & COMPARE_PARTIAL_FLAG4)) {
    var objIsWrapped = objIsObj && hasOwnProperty16.call(object, "__wrapped__"), othIsWrapped = othIsObj && hasOwnProperty16.call(other, "__wrapped__");
    if (objIsWrapped || othIsWrapped) {
      var objUnwrapped = objIsWrapped ? object.value() : object, othUnwrapped = othIsWrapped ? other.value() : other;
      stack || (stack = new Stack_default());
      return equalFunc(objUnwrapped, othUnwrapped, bitmask, customizer, stack);
    }
  }
  if (!isSameTag) {
    return false;
  }
  stack || (stack = new Stack_default());
  return equalObjects_default(object, other, bitmask, customizer, equalFunc, stack);
}
var baseIsEqualDeep_default = baseIsEqualDeep;

// node_modules/lodash-es/_baseIsEqual.js
function baseIsEqual(value, other, bitmask, customizer, stack) {
  if (value === other) {
    return true;
  }
  if (value == null || other == null || !isObjectLike_default(value) && !isObjectLike_default(other)) {
    return value !== value && other !== other;
  }
  return baseIsEqualDeep_default(value, other, bitmask, customizer, baseIsEqual, stack);
}
var baseIsEqual_default = baseIsEqual;

// node_modules/lodash-es/_baseIsMatch.js
var COMPARE_PARTIAL_FLAG5 = 1;
var COMPARE_UNORDERED_FLAG3 = 2;
function baseIsMatch(object, source, matchData, customizer) {
  var index = matchData.length, length = index, noCustomizer = !customizer;
  if (object == null) {
    return !length;
  }
  object = Object(object);
  while (index--) {
    var data = matchData[index];
    if (noCustomizer && data[2] ? data[1] !== object[data[0]] : !(data[0] in object)) {
      return false;
    }
  }
  while (++index < length) {
    data = matchData[index];
    var key = data[0], objValue = object[key], srcValue = data[1];
    if (noCustomizer && data[2]) {
      if (objValue === void 0 && !(key in object)) {
        return false;
      }
    } else {
      var stack = new Stack_default();
      if (customizer) {
        var result2 = customizer(objValue, srcValue, key, object, source, stack);
      }
      if (!(result2 === void 0 ? baseIsEqual_default(srcValue, objValue, COMPARE_PARTIAL_FLAG5 | COMPARE_UNORDERED_FLAG3, customizer, stack) : result2)) {
        return false;
      }
    }
  }
  return true;
}
var baseIsMatch_default = baseIsMatch;

// node_modules/lodash-es/_isStrictComparable.js
function isStrictComparable(value) {
  return value === value && !isObject_default(value);
}
var isStrictComparable_default = isStrictComparable;

// node_modules/lodash-es/_getMatchData.js
function getMatchData(object) {
  var result2 = keys_default(object), length = result2.length;
  while (length--) {
    var key = result2[length], value = object[key];
    result2[length] = [key, value, isStrictComparable_default(value)];
  }
  return result2;
}
var getMatchData_default = getMatchData;

// node_modules/lodash-es/_matchesStrictComparable.js
function matchesStrictComparable(key, srcValue) {
  return function(object) {
    if (object == null) {
      return false;
    }
    return object[key] === srcValue && (srcValue !== void 0 || key in Object(object));
  };
}
var matchesStrictComparable_default = matchesStrictComparable;

// node_modules/lodash-es/_baseMatches.js
function baseMatches(source) {
  var matchData = getMatchData_default(source);
  if (matchData.length == 1 && matchData[0][2]) {
    return matchesStrictComparable_default(matchData[0][0], matchData[0][1]);
  }
  return function(object) {
    return object === source || baseIsMatch_default(object, source, matchData);
  };
}
var baseMatches_default = baseMatches;

// node_modules/lodash-es/_baseHasIn.js
function baseHasIn(object, key) {
  return object != null && key in Object(object);
}
var baseHasIn_default = baseHasIn;

// node_modules/lodash-es/_hasPath.js
function hasPath(object, path, hasFunc) {
  path = castPath_default(path, object);
  var index = -1, length = path.length, result2 = false;
  while (++index < length) {
    var key = toKey_default(path[index]);
    if (!(result2 = object != null && hasFunc(object, key))) {
      break;
    }
    object = object[key];
  }
  if (result2 || ++index != length) {
    return result2;
  }
  length = object == null ? 0 : object.length;
  return !!length && isLength_default(length) && isIndex_default(key, length) && (isArray_default(object) || isArguments_default(object));
}
var hasPath_default = hasPath;

// node_modules/lodash-es/hasIn.js
function hasIn(object, path) {
  return object != null && hasPath_default(object, path, baseHasIn_default);
}
var hasIn_default = hasIn;

// node_modules/lodash-es/_baseMatchesProperty.js
var COMPARE_PARTIAL_FLAG6 = 1;
var COMPARE_UNORDERED_FLAG4 = 2;
function baseMatchesProperty(path, srcValue) {
  if (isKey_default(path) && isStrictComparable_default(srcValue)) {
    return matchesStrictComparable_default(toKey_default(path), srcValue);
  }
  return function(object) {
    var objValue = get_default(object, path);
    return objValue === void 0 && objValue === srcValue ? hasIn_default(object, path) : baseIsEqual_default(srcValue, objValue, COMPARE_PARTIAL_FLAG6 | COMPARE_UNORDERED_FLAG4);
  };
}
var baseMatchesProperty_default = baseMatchesProperty;

// node_modules/lodash-es/_baseProperty.js
function baseProperty(key) {
  return function(object) {
    return object == null ? void 0 : object[key];
  };
}
var baseProperty_default = baseProperty;

// node_modules/lodash-es/_basePropertyDeep.js
function basePropertyDeep(path) {
  return function(object) {
    return baseGet_default(object, path);
  };
}
var basePropertyDeep_default = basePropertyDeep;

// node_modules/lodash-es/property.js
function property(path) {
  return isKey_default(path) ? baseProperty_default(toKey_default(path)) : basePropertyDeep_default(path);
}
var property_default = property;

// node_modules/lodash-es/_baseIteratee.js
function baseIteratee(value) {
  if (typeof value == "function") {
    return value;
  }
  if (value == null) {
    return identity_default;
  }
  if (typeof value == "object") {
    return isArray_default(value) ? baseMatchesProperty_default(value[0], value[1]) : baseMatches_default(value);
  }
  return property_default(value);
}
var baseIteratee_default = baseIteratee;

// node_modules/lodash-es/cond.js
var FUNC_ERROR_TEXT5 = "Expected a function";
function cond(pairs) {
  var length = pairs == null ? 0 : pairs.length, toIteratee = baseIteratee_default;
  pairs = !length ? [] : arrayMap_default(pairs, function(pair) {
    if (typeof pair[1] != "function") {
      throw new TypeError(FUNC_ERROR_TEXT5);
    }
    return [toIteratee(pair[0]), pair[1]];
  });
  return baseRest_default(function(args) {
    var index = -1;
    while (++index < length) {
      var pair = pairs[index];
      if (apply_default(pair[0], this, args)) {
        return apply_default(pair[1], this, args);
      }
    }
  });
}
var cond_default = cond;

// node_modules/lodash-es/_baseConformsTo.js
function baseConformsTo(object, source, props) {
  var length = props.length;
  if (object == null) {
    return !length;
  }
  object = Object(object);
  while (length--) {
    var key = props[length], predicate = source[key], value = object[key];
    if (value === void 0 && !(key in object) || !predicate(value)) {
      return false;
    }
  }
  return true;
}
var baseConformsTo_default = baseConformsTo;

// node_modules/lodash-es/_baseConforms.js
function baseConforms(source) {
  var props = keys_default(source);
  return function(object) {
    return baseConformsTo_default(object, source, props);
  };
}
var baseConforms_default = baseConforms;

// node_modules/lodash-es/conforms.js
var CLONE_DEEP_FLAG4 = 1;
function conforms(source) {
  return baseConforms_default(baseClone_default(source, CLONE_DEEP_FLAG4));
}
var conforms_default = conforms;

// node_modules/lodash-es/conformsTo.js
function conformsTo(object, source) {
  return source == null || baseConformsTo_default(object, source, keys_default(source));
}
var conformsTo_default = conformsTo;

// node_modules/lodash-es/_arrayAggregator.js
function arrayAggregator(array, setter, iteratee2, accumulator) {
  var index = -1, length = array == null ? 0 : array.length;
  while (++index < length) {
    var value = array[index];
    setter(accumulator, value, iteratee2(value), array);
  }
  return accumulator;
}
var arrayAggregator_default = arrayAggregator;

// node_modules/lodash-es/_createBaseFor.js
function createBaseFor(fromRight) {
  return function(object, iteratee2, keysFunc) {
    var index = -1, iterable = Object(object), props = keysFunc(object), length = props.length;
    while (length--) {
      var key = props[fromRight ? length : ++index];
      if (iteratee2(iterable[key], key, iterable) === false) {
        break;
      }
    }
    return object;
  };
}
var createBaseFor_default = createBaseFor;

// node_modules/lodash-es/_baseFor.js
var baseFor = createBaseFor_default();
var baseFor_default = baseFor;

// node_modules/lodash-es/_baseForOwn.js
function baseForOwn(object, iteratee2) {
  return object && baseFor_default(object, iteratee2, keys_default);
}
var baseForOwn_default = baseForOwn;

// node_modules/lodash-es/_createBaseEach.js
function createBaseEach(eachFunc, fromRight) {
  return function(collection, iteratee2) {
    if (collection == null) {
      return collection;
    }
    if (!isArrayLike_default(collection)) {
      return eachFunc(collection, iteratee2);
    }
    var length = collection.length, index = fromRight ? length : -1, iterable = Object(collection);
    while (fromRight ? index-- : ++index < length) {
      if (iteratee2(iterable[index], index, iterable) === false) {
        break;
      }
    }
    return collection;
  };
}
var createBaseEach_default = createBaseEach;

// node_modules/lodash-es/_baseEach.js
var baseEach = createBaseEach_default(baseForOwn_default);
var baseEach_default = baseEach;

// node_modules/lodash-es/_baseAggregator.js
function baseAggregator(collection, setter, iteratee2, accumulator) {
  baseEach_default(collection, function(value, key, collection2) {
    setter(accumulator, value, iteratee2(value), collection2);
  });
  return accumulator;
}
var baseAggregator_default = baseAggregator;

// node_modules/lodash-es/_createAggregator.js
function createAggregator(setter, initializer) {
  return function(collection, iteratee2) {
    var func = isArray_default(collection) ? arrayAggregator_default : baseAggregator_default, accumulator = initializer ? initializer() : {};
    return func(collection, setter, baseIteratee_default(iteratee2, 2), accumulator);
  };
}
var createAggregator_default = createAggregator;

// node_modules/lodash-es/countBy.js
var objectProto20 = Object.prototype;
var hasOwnProperty17 = objectProto20.hasOwnProperty;
var countBy = createAggregator_default(function(result2, value, key) {
  if (hasOwnProperty17.call(result2, key)) {
    ++result2[key];
  } else {
    baseAssignValue_default(result2, key, 1);
  }
});
var countBy_default = countBy;

// node_modules/lodash-es/create.js
function create(prototype, properties) {
  var result2 = baseCreate_default(prototype);
  return properties == null ? result2 : baseAssign_default(result2, properties);
}
var create_default = create;

// node_modules/lodash-es/curry.js
var WRAP_CURRY_FLAG6 = 8;
function curry(func, arity, guard) {
  arity = guard ? void 0 : arity;
  var result2 = createWrap_default(func, WRAP_CURRY_FLAG6, void 0, void 0, void 0, void 0, void 0, arity);
  result2.placeholder = curry.placeholder;
  return result2;
}
curry.placeholder = {};
var curry_default = curry;

// node_modules/lodash-es/curryRight.js
var WRAP_CURRY_RIGHT_FLAG4 = 16;
function curryRight(func, arity, guard) {
  arity = guard ? void 0 : arity;
  var result2 = createWrap_default(func, WRAP_CURRY_RIGHT_FLAG4, void 0, void 0, void 0, void 0, void 0, arity);
  result2.placeholder = curryRight.placeholder;
  return result2;
}
curryRight.placeholder = {};
var curryRight_default = curryRight;

// node_modules/lodash-es/now.js
var now = function() {
  return root_default.Date.now();
};
var now_default = now;

// node_modules/lodash-es/debounce.js
var FUNC_ERROR_TEXT6 = "Expected a function";
var nativeMax6 = Math.max;
var nativeMin4 = Math.min;
function debounce(func, wait, options) {
  var lastArgs, lastThis, maxWait, result2, timerId, lastCallTime, lastInvokeTime = 0, leading = false, maxing = false, trailing = true;
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT6);
  }
  wait = toNumber_default(wait) || 0;
  if (isObject_default(options)) {
    leading = !!options.leading;
    maxing = "maxWait" in options;
    maxWait = maxing ? nativeMax6(toNumber_default(options.maxWait) || 0, wait) : maxWait;
    trailing = "trailing" in options ? !!options.trailing : trailing;
  }
  function invokeFunc(time) {
    var args = lastArgs, thisArg = lastThis;
    lastArgs = lastThis = void 0;
    lastInvokeTime = time;
    result2 = func.apply(thisArg, args);
    return result2;
  }
  function leadingEdge(time) {
    lastInvokeTime = time;
    timerId = setTimeout(timerExpired, wait);
    return leading ? invokeFunc(time) : result2;
  }
  function remainingWait(time) {
    var timeSinceLastCall = time - lastCallTime, timeSinceLastInvoke = time - lastInvokeTime, timeWaiting = wait - timeSinceLastCall;
    return maxing ? nativeMin4(timeWaiting, maxWait - timeSinceLastInvoke) : timeWaiting;
  }
  function shouldInvoke(time) {
    var timeSinceLastCall = time - lastCallTime, timeSinceLastInvoke = time - lastInvokeTime;
    return lastCallTime === void 0 || timeSinceLastCall >= wait || timeSinceLastCall < 0 || maxing && timeSinceLastInvoke >= maxWait;
  }
  function timerExpired() {
    var time = now_default();
    if (shouldInvoke(time)) {
      return trailingEdge(time);
    }
    timerId = setTimeout(timerExpired, remainingWait(time));
  }
  function trailingEdge(time) {
    timerId = void 0;
    if (trailing && lastArgs) {
      return invokeFunc(time);
    }
    lastArgs = lastThis = void 0;
    return result2;
  }
  function cancel() {
    if (timerId !== void 0) {
      clearTimeout(timerId);
    }
    lastInvokeTime = 0;
    lastArgs = lastCallTime = lastThis = timerId = void 0;
  }
  function flush() {
    return timerId === void 0 ? result2 : trailingEdge(now_default());
  }
  function debounced() {
    var time = now_default(), isInvoking = shouldInvoke(time);
    lastArgs = arguments;
    lastThis = this;
    lastCallTime = time;
    if (isInvoking) {
      if (timerId === void 0) {
        return leadingEdge(lastCallTime);
      }
      if (maxing) {
        clearTimeout(timerId);
        timerId = setTimeout(timerExpired, wait);
        return invokeFunc(lastCallTime);
      }
    }
    if (timerId === void 0) {
      timerId = setTimeout(timerExpired, wait);
    }
    return result2;
  }
  debounced.cancel = cancel;
  debounced.flush = flush;
  return debounced;
}
var debounce_default = debounce;

// node_modules/lodash-es/defaultTo.js
function defaultTo(value, defaultValue) {
  return value == null || value !== value ? defaultValue : value;
}
var defaultTo_default = defaultTo;

// node_modules/lodash-es/defaults.js
var objectProto21 = Object.prototype;
var hasOwnProperty18 = objectProto21.hasOwnProperty;
var defaults = baseRest_default(function(object, sources) {
  object = Object(object);
  var index = -1;
  var length = sources.length;
  var guard = length > 2 ? sources[2] : void 0;
  if (guard && isIterateeCall_default(sources[0], sources[1], guard)) {
    length = 1;
  }
  while (++index < length) {
    var source = sources[index];
    var props = keysIn_default(source);
    var propsIndex = -1;
    var propsLength = props.length;
    while (++propsIndex < propsLength) {
      var key = props[propsIndex];
      var value = object[key];
      if (value === void 0 || eq_default(value, objectProto21[key]) && !hasOwnProperty18.call(object, key)) {
        object[key] = source[key];
      }
    }
  }
  return object;
});
var defaults_default = defaults;

// node_modules/lodash-es/_assignMergeValue.js
function assignMergeValue(object, key, value) {
  if (value !== void 0 && !eq_default(object[key], value) || value === void 0 && !(key in object)) {
    baseAssignValue_default(object, key, value);
  }
}
var assignMergeValue_default = assignMergeValue;

// node_modules/lodash-es/isArrayLikeObject.js
function isArrayLikeObject(value) {
  return isObjectLike_default(value) && isArrayLike_default(value);
}
var isArrayLikeObject_default = isArrayLikeObject;

// node_modules/lodash-es/_safeGet.js
function safeGet(object, key) {
  if (key === "constructor" && typeof object[key] === "function") {
    return;
  }
  if (key == "__proto__") {
    return;
  }
  return object[key];
}
var safeGet_default = safeGet;

// node_modules/lodash-es/toPlainObject.js
function toPlainObject(value) {
  return copyObject_default(value, keysIn_default(value));
}
var toPlainObject_default = toPlainObject;

// node_modules/lodash-es/_baseMergeDeep.js
function baseMergeDeep(object, source, key, srcIndex, mergeFunc, customizer, stack) {
  var objValue = safeGet_default(object, key), srcValue = safeGet_default(source, key), stacked = stack.get(srcValue);
  if (stacked) {
    assignMergeValue_default(object, key, stacked);
    return;
  }
  var newValue = customizer ? customizer(objValue, srcValue, key + "", object, source, stack) : void 0;
  var isCommon = newValue === void 0;
  if (isCommon) {
    var isArr = isArray_default(srcValue), isBuff = !isArr && isBuffer_default(srcValue), isTyped = !isArr && !isBuff && isTypedArray_default(srcValue);
    newValue = srcValue;
    if (isArr || isBuff || isTyped) {
      if (isArray_default(objValue)) {
        newValue = objValue;
      } else if (isArrayLikeObject_default(objValue)) {
        newValue = copyArray_default(objValue);
      } else if (isBuff) {
        isCommon = false;
        newValue = cloneBuffer_default(srcValue, true);
      } else if (isTyped) {
        isCommon = false;
        newValue = cloneTypedArray_default(srcValue, true);
      } else {
        newValue = [];
      }
    } else if (isPlainObject_default(srcValue) || isArguments_default(srcValue)) {
      newValue = objValue;
      if (isArguments_default(objValue)) {
        newValue = toPlainObject_default(objValue);
      } else if (!isObject_default(objValue) || isFunction_default(objValue)) {
        newValue = initCloneObject_default(srcValue);
      }
    } else {
      isCommon = false;
    }
  }
  if (isCommon) {
    stack.set(srcValue, newValue);
    mergeFunc(newValue, srcValue, srcIndex, customizer, stack);
    stack["delete"](srcValue);
  }
  assignMergeValue_default(object, key, newValue);
}
var baseMergeDeep_default = baseMergeDeep;

// node_modules/lodash-es/_baseMerge.js
function baseMerge(object, source, srcIndex, customizer, stack) {
  if (object === source) {
    return;
  }
  baseFor_default(source, function(srcValue, key) {
    stack || (stack = new Stack_default());
    if (isObject_default(srcValue)) {
      baseMergeDeep_default(object, source, key, srcIndex, baseMerge, customizer, stack);
    } else {
      var newValue = customizer ? customizer(safeGet_default(object, key), srcValue, key + "", object, source, stack) : void 0;
      if (newValue === void 0) {
        newValue = srcValue;
      }
      assignMergeValue_default(object, key, newValue);
    }
  }, keysIn_default);
}
var baseMerge_default = baseMerge;

// node_modules/lodash-es/_customDefaultsMerge.js
function customDefaultsMerge(objValue, srcValue, key, object, source, stack) {
  if (isObject_default(objValue) && isObject_default(srcValue)) {
    stack.set(srcValue, objValue);
    baseMerge_default(objValue, srcValue, void 0, customDefaultsMerge, stack);
    stack["delete"](srcValue);
  }
  return objValue;
}
var customDefaultsMerge_default = customDefaultsMerge;

// node_modules/lodash-es/mergeWith.js
var mergeWith = createAssigner_default(function(object, source, srcIndex, customizer) {
  baseMerge_default(object, source, srcIndex, customizer);
});
var mergeWith_default = mergeWith;

// node_modules/lodash-es/defaultsDeep.js
var defaultsDeep = baseRest_default(function(args) {
  args.push(void 0, customDefaultsMerge_default);
  return apply_default(mergeWith_default, void 0, args);
});
var defaultsDeep_default = defaultsDeep;

// node_modules/lodash-es/_baseDelay.js
var FUNC_ERROR_TEXT7 = "Expected a function";
function baseDelay(func, wait, args) {
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT7);
  }
  return setTimeout(function() {
    func.apply(void 0, args);
  }, wait);
}
var baseDelay_default = baseDelay;

// node_modules/lodash-es/defer.js
var defer = baseRest_default(function(func, args) {
  return baseDelay_default(func, 1, args);
});
var defer_default = defer;

// node_modules/lodash-es/delay.js
var delay = baseRest_default(function(func, wait, args) {
  return baseDelay_default(func, toNumber_default(wait) || 0, args);
});
var delay_default = delay;

// node_modules/lodash-es/_arrayIncludesWith.js
function arrayIncludesWith(array, value, comparator) {
  var index = -1, length = array == null ? 0 : array.length;
  while (++index < length) {
    if (comparator(value, array[index])) {
      return true;
    }
  }
  return false;
}
var arrayIncludesWith_default = arrayIncludesWith;

// node_modules/lodash-es/_baseDifference.js
var LARGE_ARRAY_SIZE2 = 200;
function baseDifference(array, values2, iteratee2, comparator) {
  var index = -1, includes2 = arrayIncludes_default, isCommon = true, length = array.length, result2 = [], valuesLength = values2.length;
  if (!length) {
    return result2;
  }
  if (iteratee2) {
    values2 = arrayMap_default(values2, baseUnary_default(iteratee2));
  }
  if (comparator) {
    includes2 = arrayIncludesWith_default;
    isCommon = false;
  } else if (values2.length >= LARGE_ARRAY_SIZE2) {
    includes2 = cacheHas_default;
    isCommon = false;
    values2 = new SetCache_default(values2);
  }
  outer:
    while (++index < length) {
      var value = array[index], computed = iteratee2 == null ? value : iteratee2(value);
      value = comparator || value !== 0 ? value : 0;
      if (isCommon && computed === computed) {
        var valuesIndex = valuesLength;
        while (valuesIndex--) {
          if (values2[valuesIndex] === computed) {
            continue outer;
          }
        }
        result2.push(value);
      } else if (!includes2(values2, computed, comparator)) {
        result2.push(value);
      }
    }
  return result2;
}
var baseDifference_default = baseDifference;

// node_modules/lodash-es/difference.js
var difference = baseRest_default(function(array, values2) {
  return isArrayLikeObject_default(array) ? baseDifference_default(array, baseFlatten_default(values2, 1, isArrayLikeObject_default, true)) : [];
});
var difference_default = difference;

// node_modules/lodash-es/last.js
function last(array) {
  var length = array == null ? 0 : array.length;
  return length ? array[length - 1] : void 0;
}
var last_default = last;

// node_modules/lodash-es/differenceBy.js
var differenceBy = baseRest_default(function(array, values2) {
  var iteratee2 = last_default(values2);
  if (isArrayLikeObject_default(iteratee2)) {
    iteratee2 = void 0;
  }
  return isArrayLikeObject_default(array) ? baseDifference_default(array, baseFlatten_default(values2, 1, isArrayLikeObject_default, true), baseIteratee_default(iteratee2, 2)) : [];
});
var differenceBy_default = differenceBy;

// node_modules/lodash-es/differenceWith.js
var differenceWith = baseRest_default(function(array, values2) {
  var comparator = last_default(values2);
  if (isArrayLikeObject_default(comparator)) {
    comparator = void 0;
  }
  return isArrayLikeObject_default(array) ? baseDifference_default(array, baseFlatten_default(values2, 1, isArrayLikeObject_default, true), void 0, comparator) : [];
});
var differenceWith_default = differenceWith;

// node_modules/lodash-es/divide.js
var divide = createMathOperation_default(function(dividend, divisor) {
  return dividend / divisor;
}, 1);
var divide_default = divide;

// node_modules/lodash-es/drop.js
function drop(array, n, guard) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return [];
  }
  n = guard || n === void 0 ? 1 : toInteger_default(n);
  return baseSlice_default(array, n < 0 ? 0 : n, length);
}
var drop_default = drop;

// node_modules/lodash-es/dropRight.js
function dropRight(array, n, guard) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return [];
  }
  n = guard || n === void 0 ? 1 : toInteger_default(n);
  n = length - n;
  return baseSlice_default(array, 0, n < 0 ? 0 : n);
}
var dropRight_default = dropRight;

// node_modules/lodash-es/_baseWhile.js
function baseWhile(array, predicate, isDrop, fromRight) {
  var length = array.length, index = fromRight ? length : -1;
  while ((fromRight ? index-- : ++index < length) && predicate(array[index], index, array)) {
  }
  return isDrop ? baseSlice_default(array, fromRight ? 0 : index, fromRight ? index + 1 : length) : baseSlice_default(array, fromRight ? index + 1 : 0, fromRight ? length : index);
}
var baseWhile_default = baseWhile;

// node_modules/lodash-es/dropRightWhile.js
function dropRightWhile(array, predicate) {
  return array && array.length ? baseWhile_default(array, baseIteratee_default(predicate, 3), true, true) : [];
}
var dropRightWhile_default = dropRightWhile;

// node_modules/lodash-es/dropWhile.js
function dropWhile(array, predicate) {
  return array && array.length ? baseWhile_default(array, baseIteratee_default(predicate, 3), true) : [];
}
var dropWhile_default = dropWhile;

// node_modules/lodash-es/_castFunction.js
function castFunction(value) {
  return typeof value == "function" ? value : identity_default;
}
var castFunction_default = castFunction;

// node_modules/lodash-es/forEach.js
function forEach(collection, iteratee2) {
  var func = isArray_default(collection) ? arrayEach_default : baseEach_default;
  return func(collection, castFunction_default(iteratee2));
}
var forEach_default = forEach;

// node_modules/lodash-es/_arrayEachRight.js
function arrayEachRight(array, iteratee2) {
  var length = array == null ? 0 : array.length;
  while (length--) {
    if (iteratee2(array[length], length, array) === false) {
      break;
    }
  }
  return array;
}
var arrayEachRight_default = arrayEachRight;

// node_modules/lodash-es/_baseForRight.js
var baseForRight = createBaseFor_default(true);
var baseForRight_default = baseForRight;

// node_modules/lodash-es/_baseForOwnRight.js
function baseForOwnRight(object, iteratee2) {
  return object && baseForRight_default(object, iteratee2, keys_default);
}
var baseForOwnRight_default = baseForOwnRight;

// node_modules/lodash-es/_baseEachRight.js
var baseEachRight = createBaseEach_default(baseForOwnRight_default, true);
var baseEachRight_default = baseEachRight;

// node_modules/lodash-es/forEachRight.js
function forEachRight(collection, iteratee2) {
  var func = isArray_default(collection) ? arrayEachRight_default : baseEachRight_default;
  return func(collection, castFunction_default(iteratee2));
}
var forEachRight_default = forEachRight;

// node_modules/lodash-es/endsWith.js
function endsWith(string, target, position) {
  string = toString_default(string);
  target = baseToString_default(target);
  var length = string.length;
  position = position === void 0 ? length : baseClamp_default(toInteger_default(position), 0, length);
  var end = position;
  position -= target.length;
  return position >= 0 && string.slice(position, end) == target;
}
var endsWith_default = endsWith;

// node_modules/lodash-es/_baseToPairs.js
function baseToPairs(object, props) {
  return arrayMap_default(props, function(key) {
    return [key, object[key]];
  });
}
var baseToPairs_default = baseToPairs;

// node_modules/lodash-es/_setToPairs.js
function setToPairs(set2) {
  var index = -1, result2 = Array(set2.size);
  set2.forEach(function(value) {
    result2[++index] = [value, value];
  });
  return result2;
}
var setToPairs_default = setToPairs;

// node_modules/lodash-es/_createToPairs.js
var mapTag7 = "[object Map]";
var setTag7 = "[object Set]";
function createToPairs(keysFunc) {
  return function(object) {
    var tag = getTag_default(object);
    if (tag == mapTag7) {
      return mapToArray_default(object);
    }
    if (tag == setTag7) {
      return setToPairs_default(object);
    }
    return baseToPairs_default(object, keysFunc(object));
  };
}
var createToPairs_default = createToPairs;

// node_modules/lodash-es/toPairs.js
var toPairs = createToPairs_default(keys_default);
var toPairs_default = toPairs;

// node_modules/lodash-es/toPairsIn.js
var toPairsIn = createToPairs_default(keysIn_default);
var toPairsIn_default = toPairsIn;

// node_modules/lodash-es/_escapeHtmlChar.js
var htmlEscapes = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
};
var escapeHtmlChar = basePropertyOf_default(htmlEscapes);
var escapeHtmlChar_default = escapeHtmlChar;

// node_modules/lodash-es/escape.js
var reUnescapedHtml = /[&<>"']/g;
var reHasUnescapedHtml = RegExp(reUnescapedHtml.source);
function escape(string) {
  string = toString_default(string);
  return string && reHasUnescapedHtml.test(string) ? string.replace(reUnescapedHtml, escapeHtmlChar_default) : string;
}
var escape_default = escape;

// node_modules/lodash-es/escapeRegExp.js
var reRegExpChar2 = /[\\^$.*+?()[\]{}|]/g;
var reHasRegExpChar = RegExp(reRegExpChar2.source);
function escapeRegExp2(string) {
  string = toString_default(string);
  return string && reHasRegExpChar.test(string) ? string.replace(reRegExpChar2, "\\$&") : string;
}
var escapeRegExp_default = escapeRegExp2;

// node_modules/lodash-es/_arrayEvery.js
function arrayEvery(array, predicate) {
  var index = -1, length = array == null ? 0 : array.length;
  while (++index < length) {
    if (!predicate(array[index], index, array)) {
      return false;
    }
  }
  return true;
}
var arrayEvery_default = arrayEvery;

// node_modules/lodash-es/_baseEvery.js
function baseEvery(collection, predicate) {
  var result2 = true;
  baseEach_default(collection, function(value, index, collection2) {
    result2 = !!predicate(value, index, collection2);
    return result2;
  });
  return result2;
}
var baseEvery_default = baseEvery;

// node_modules/lodash-es/every.js
function every(collection, predicate, guard) {
  var func = isArray_default(collection) ? arrayEvery_default : baseEvery_default;
  if (guard && isIterateeCall_default(collection, predicate, guard)) {
    predicate = void 0;
  }
  return func(collection, baseIteratee_default(predicate, 3));
}
var every_default = every;

// node_modules/lodash-es/toLength.js
var MAX_ARRAY_LENGTH2 = 4294967295;
function toLength(value) {
  return value ? baseClamp_default(toInteger_default(value), 0, MAX_ARRAY_LENGTH2) : 0;
}
var toLength_default = toLength;

// node_modules/lodash-es/_baseFill.js
function baseFill(array, value, start, end) {
  var length = array.length;
  start = toInteger_default(start);
  if (start < 0) {
    start = -start > length ? 0 : length + start;
  }
  end = end === void 0 || end > length ? length : toInteger_default(end);
  if (end < 0) {
    end += length;
  }
  end = start > end ? 0 : toLength_default(end);
  while (start < end) {
    array[start++] = value;
  }
  return array;
}
var baseFill_default = baseFill;

// node_modules/lodash-es/fill.js
function fill(array, value, start, end) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return [];
  }
  if (start && typeof start != "number" && isIterateeCall_default(array, value, start)) {
    start = 0;
    end = length;
  }
  return baseFill_default(array, value, start, end);
}
var fill_default = fill;

// node_modules/lodash-es/_baseFilter.js
function baseFilter(collection, predicate) {
  var result2 = [];
  baseEach_default(collection, function(value, index, collection2) {
    if (predicate(value, index, collection2)) {
      result2.push(value);
    }
  });
  return result2;
}
var baseFilter_default = baseFilter;

// node_modules/lodash-es/filter.js
function filter(collection, predicate) {
  var func = isArray_default(collection) ? arrayFilter_default : baseFilter_default;
  return func(collection, baseIteratee_default(predicate, 3));
}
var filter_default = filter;

// node_modules/lodash-es/_createFind.js
function createFind(findIndexFunc) {
  return function(collection, predicate, fromIndex) {
    var iterable = Object(collection);
    if (!isArrayLike_default(collection)) {
      var iteratee2 = baseIteratee_default(predicate, 3);
      collection = keys_default(collection);
      predicate = function(key) {
        return iteratee2(iterable[key], key, iterable);
      };
    }
    var index = findIndexFunc(collection, predicate, fromIndex);
    return index > -1 ? iterable[iteratee2 ? collection[index] : index] : void 0;
  };
}
var createFind_default = createFind;

// node_modules/lodash-es/findIndex.js
var nativeMax7 = Math.max;
function findIndex(array, predicate, fromIndex) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return -1;
  }
  var index = fromIndex == null ? 0 : toInteger_default(fromIndex);
  if (index < 0) {
    index = nativeMax7(length + index, 0);
  }
  return baseFindIndex_default(array, baseIteratee_default(predicate, 3), index);
}
var findIndex_default = findIndex;

// node_modules/lodash-es/find.js
var find = createFind_default(findIndex_default);
var find_default = find;

// node_modules/lodash-es/_baseFindKey.js
function baseFindKey(collection, predicate, eachFunc) {
  var result2;
  eachFunc(collection, function(value, key, collection2) {
    if (predicate(value, key, collection2)) {
      result2 = key;
      return false;
    }
  });
  return result2;
}
var baseFindKey_default = baseFindKey;

// node_modules/lodash-es/findKey.js
function findKey(object, predicate) {
  return baseFindKey_default(object, baseIteratee_default(predicate, 3), baseForOwn_default);
}
var findKey_default = findKey;

// node_modules/lodash-es/findLastIndex.js
var nativeMax8 = Math.max;
var nativeMin5 = Math.min;
function findLastIndex(array, predicate, fromIndex) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return -1;
  }
  var index = length - 1;
  if (fromIndex !== void 0) {
    index = toInteger_default(fromIndex);
    index = fromIndex < 0 ? nativeMax8(length + index, 0) : nativeMin5(index, length - 1);
  }
  return baseFindIndex_default(array, baseIteratee_default(predicate, 3), index, true);
}
var findLastIndex_default = findLastIndex;

// node_modules/lodash-es/findLast.js
var findLast = createFind_default(findLastIndex_default);
var findLast_default = findLast;

// node_modules/lodash-es/findLastKey.js
function findLastKey(object, predicate) {
  return baseFindKey_default(object, baseIteratee_default(predicate, 3), baseForOwnRight_default);
}
var findLastKey_default = findLastKey;

// node_modules/lodash-es/head.js
function head(array) {
  return array && array.length ? array[0] : void 0;
}
var head_default = head;

// node_modules/lodash-es/_baseMap.js
function baseMap(collection, iteratee2) {
  var index = -1, result2 = isArrayLike_default(collection) ? Array(collection.length) : [];
  baseEach_default(collection, function(value, key, collection2) {
    result2[++index] = iteratee2(value, key, collection2);
  });
  return result2;
}
var baseMap_default = baseMap;

// node_modules/lodash-es/map.js
function map(collection, iteratee2) {
  var func = isArray_default(collection) ? arrayMap_default : baseMap_default;
  return func(collection, baseIteratee_default(iteratee2, 3));
}
var map_default = map;

// node_modules/lodash-es/flatMap.js
function flatMap(collection, iteratee2) {
  return baseFlatten_default(map_default(collection, iteratee2), 1);
}
var flatMap_default = flatMap;

// node_modules/lodash-es/flatMapDeep.js
var INFINITY4 = 1 / 0;
function flatMapDeep(collection, iteratee2) {
  return baseFlatten_default(map_default(collection, iteratee2), INFINITY4);
}
var flatMapDeep_default = flatMapDeep;

// node_modules/lodash-es/flatMapDepth.js
function flatMapDepth(collection, iteratee2, depth) {
  depth = depth === void 0 ? 1 : toInteger_default(depth);
  return baseFlatten_default(map_default(collection, iteratee2), depth);
}
var flatMapDepth_default = flatMapDepth;

// node_modules/lodash-es/flattenDeep.js
var INFINITY5 = 1 / 0;
function flattenDeep(array) {
  var length = array == null ? 0 : array.length;
  return length ? baseFlatten_default(array, INFINITY5) : [];
}
var flattenDeep_default = flattenDeep;

// node_modules/lodash-es/flattenDepth.js
function flattenDepth(array, depth) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return [];
  }
  depth = depth === void 0 ? 1 : toInteger_default(depth);
  return baseFlatten_default(array, depth);
}
var flattenDepth_default = flattenDepth;

// node_modules/lodash-es/flip.js
var WRAP_FLIP_FLAG3 = 512;
function flip(func) {
  return createWrap_default(func, WRAP_FLIP_FLAG3);
}
var flip_default = flip;

// node_modules/lodash-es/floor.js
var floor = createRound_default("floor");
var floor_default = floor;

// node_modules/lodash-es/_createFlow.js
var FUNC_ERROR_TEXT8 = "Expected a function";
var WRAP_CURRY_FLAG7 = 8;
var WRAP_PARTIAL_FLAG6 = 32;
var WRAP_ARY_FLAG5 = 128;
var WRAP_REARG_FLAG3 = 256;
function createFlow(fromRight) {
  return flatRest_default(function(funcs) {
    var length = funcs.length, index = length, prereq = LodashWrapper_default.prototype.thru;
    if (fromRight) {
      funcs.reverse();
    }
    while (index--) {
      var func = funcs[index];
      if (typeof func != "function") {
        throw new TypeError(FUNC_ERROR_TEXT8);
      }
      if (prereq && !wrapper && getFuncName_default(func) == "wrapper") {
        var wrapper = new LodashWrapper_default([], true);
      }
    }
    index = wrapper ? index : length;
    while (++index < length) {
      func = funcs[index];
      var funcName = getFuncName_default(func), data = funcName == "wrapper" ? getData_default(func) : void 0;
      if (data && isLaziable_default(data[0]) && data[1] == (WRAP_ARY_FLAG5 | WRAP_CURRY_FLAG7 | WRAP_PARTIAL_FLAG6 | WRAP_REARG_FLAG3) && !data[4].length && data[9] == 1) {
        wrapper = wrapper[getFuncName_default(data[0])].apply(wrapper, data[3]);
      } else {
        wrapper = func.length == 1 && isLaziable_default(func) ? wrapper[funcName]() : wrapper.thru(func);
      }
    }
    return function() {
      var args = arguments, value = args[0];
      if (wrapper && args.length == 1 && isArray_default(value)) {
        return wrapper.plant(value).value();
      }
      var index2 = 0, result2 = length ? funcs[index2].apply(this, args) : value;
      while (++index2 < length) {
        result2 = funcs[index2].call(this, result2);
      }
      return result2;
    };
  });
}
var createFlow_default = createFlow;

// node_modules/lodash-es/flow.js
var flow = createFlow_default();
var flow_default = flow;

// node_modules/lodash-es/flowRight.js
var flowRight = createFlow_default(true);
var flowRight_default = flowRight;

// node_modules/lodash-es/forIn.js
function forIn(object, iteratee2) {
  return object == null ? object : baseFor_default(object, castFunction_default(iteratee2), keysIn_default);
}
var forIn_default = forIn;

// node_modules/lodash-es/forInRight.js
function forInRight(object, iteratee2) {
  return object == null ? object : baseForRight_default(object, castFunction_default(iteratee2), keysIn_default);
}
var forInRight_default = forInRight;

// node_modules/lodash-es/forOwn.js
function forOwn(object, iteratee2) {
  return object && baseForOwn_default(object, castFunction_default(iteratee2));
}
var forOwn_default = forOwn;

// node_modules/lodash-es/forOwnRight.js
function forOwnRight(object, iteratee2) {
  return object && baseForOwnRight_default(object, castFunction_default(iteratee2));
}
var forOwnRight_default = forOwnRight;

// node_modules/lodash-es/fromPairs.js
function fromPairs(pairs) {
  var index = -1, length = pairs == null ? 0 : pairs.length, result2 = {};
  while (++index < length) {
    var pair = pairs[index];
    result2[pair[0]] = pair[1];
  }
  return result2;
}
var fromPairs_default = fromPairs;

// node_modules/lodash-es/_baseFunctions.js
function baseFunctions(object, props) {
  return arrayFilter_default(props, function(key) {
    return isFunction_default(object[key]);
  });
}
var baseFunctions_default = baseFunctions;

// node_modules/lodash-es/functions.js
function functions(object) {
  return object == null ? [] : baseFunctions_default(object, keys_default(object));
}
var functions_default = functions;

// node_modules/lodash-es/functionsIn.js
function functionsIn(object) {
  return object == null ? [] : baseFunctions_default(object, keysIn_default(object));
}
var functionsIn_default = functionsIn;

// node_modules/lodash-es/groupBy.js
var objectProto22 = Object.prototype;
var hasOwnProperty19 = objectProto22.hasOwnProperty;
var groupBy = createAggregator_default(function(result2, value, key) {
  if (hasOwnProperty19.call(result2, key)) {
    result2[key].push(value);
  } else {
    baseAssignValue_default(result2, key, [value]);
  }
});
var groupBy_default = groupBy;

// node_modules/lodash-es/_baseGt.js
function baseGt(value, other) {
  return value > other;
}
var baseGt_default = baseGt;

// node_modules/lodash-es/_createRelationalOperation.js
function createRelationalOperation(operator) {
  return function(value, other) {
    if (!(typeof value == "string" && typeof other == "string")) {
      value = toNumber_default(value);
      other = toNumber_default(other);
    }
    return operator(value, other);
  };
}
var createRelationalOperation_default = createRelationalOperation;

// node_modules/lodash-es/gt.js
var gt = createRelationalOperation_default(baseGt_default);
var gt_default = gt;

// node_modules/lodash-es/gte.js
var gte = createRelationalOperation_default(function(value, other) {
  return value >= other;
});
var gte_default = gte;

// node_modules/lodash-es/_baseHas.js
var objectProto23 = Object.prototype;
var hasOwnProperty20 = objectProto23.hasOwnProperty;
function baseHas(object, key) {
  return object != null && hasOwnProperty20.call(object, key);
}
var baseHas_default = baseHas;

// node_modules/lodash-es/has.js
function has(object, path) {
  return object != null && hasPath_default(object, path, baseHas_default);
}
var has_default = has;

// node_modules/lodash-es/_baseInRange.js
var nativeMax9 = Math.max;
var nativeMin6 = Math.min;
function baseInRange(number, start, end) {
  return number >= nativeMin6(start, end) && number < nativeMax9(start, end);
}
var baseInRange_default = baseInRange;

// node_modules/lodash-es/inRange.js
function inRange2(number, start, end) {
  start = toFinite_default(start);
  if (end === void 0) {
    end = start;
    start = 0;
  } else {
    end = toFinite_default(end);
  }
  number = toNumber_default(number);
  return baseInRange_default(number, start, end);
}
var inRange_default = inRange2;

// node_modules/lodash-es/isString.js
var stringTag5 = "[object String]";
function isString(value) {
  return typeof value == "string" || !isArray_default(value) && isObjectLike_default(value) && baseGetTag_default(value) == stringTag5;
}
var isString_default = isString;

// node_modules/lodash-es/_baseValues.js
function baseValues(object, props) {
  return arrayMap_default(props, function(key) {
    return object[key];
  });
}
var baseValues_default = baseValues;

// node_modules/lodash-es/values.js
function values(object) {
  return object == null ? [] : baseValues_default(object, keys_default(object));
}
var values_default = values;

// node_modules/lodash-es/includes.js
var nativeMax10 = Math.max;
function includes(collection, value, fromIndex, guard) {
  collection = isArrayLike_default(collection) ? collection : values_default(collection);
  fromIndex = fromIndex && !guard ? toInteger_default(fromIndex) : 0;
  var length = collection.length;
  if (fromIndex < 0) {
    fromIndex = nativeMax10(length + fromIndex, 0);
  }
  return isString_default(collection) ? fromIndex <= length && collection.indexOf(value, fromIndex) > -1 : !!length && baseIndexOf_default(collection, value, fromIndex) > -1;
}
var includes_default = includes;

// node_modules/lodash-es/indexOf.js
var nativeMax11 = Math.max;
function indexOf(array, value, fromIndex) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return -1;
  }
  var index = fromIndex == null ? 0 : toInteger_default(fromIndex);
  if (index < 0) {
    index = nativeMax11(length + index, 0);
  }
  return baseIndexOf_default(array, value, index);
}
var indexOf_default = indexOf;

// node_modules/lodash-es/initial.js
function initial(array) {
  var length = array == null ? 0 : array.length;
  return length ? baseSlice_default(array, 0, -1) : [];
}
var initial_default = initial;

// node_modules/lodash-es/_baseIntersection.js
var nativeMin7 = Math.min;
function baseIntersection(arrays, iteratee2, comparator) {
  var includes2 = comparator ? arrayIncludesWith_default : arrayIncludes_default, length = arrays[0].length, othLength = arrays.length, othIndex = othLength, caches = Array(othLength), maxLength = Infinity, result2 = [];
  while (othIndex--) {
    var array = arrays[othIndex];
    if (othIndex && iteratee2) {
      array = arrayMap_default(array, baseUnary_default(iteratee2));
    }
    maxLength = nativeMin7(array.length, maxLength);
    caches[othIndex] = !comparator && (iteratee2 || length >= 120 && array.length >= 120) ? new SetCache_default(othIndex && array) : void 0;
  }
  array = arrays[0];
  var index = -1, seen = caches[0];
  outer:
    while (++index < length && result2.length < maxLength) {
      var value = array[index], computed = iteratee2 ? iteratee2(value) : value;
      value = comparator || value !== 0 ? value : 0;
      if (!(seen ? cacheHas_default(seen, computed) : includes2(result2, computed, comparator))) {
        othIndex = othLength;
        while (--othIndex) {
          var cache = caches[othIndex];
          if (!(cache ? cacheHas_default(cache, computed) : includes2(arrays[othIndex], computed, comparator))) {
            continue outer;
          }
        }
        if (seen) {
          seen.push(computed);
        }
        result2.push(value);
      }
    }
  return result2;
}
var baseIntersection_default = baseIntersection;

// node_modules/lodash-es/_castArrayLikeObject.js
function castArrayLikeObject(value) {
  return isArrayLikeObject_default(value) ? value : [];
}
var castArrayLikeObject_default = castArrayLikeObject;

// node_modules/lodash-es/intersection.js
var intersection = baseRest_default(function(arrays) {
  var mapped = arrayMap_default(arrays, castArrayLikeObject_default);
  return mapped.length && mapped[0] === arrays[0] ? baseIntersection_default(mapped) : [];
});
var intersection_default = intersection;

// node_modules/lodash-es/intersectionBy.js
var intersectionBy = baseRest_default(function(arrays) {
  var iteratee2 = last_default(arrays), mapped = arrayMap_default(arrays, castArrayLikeObject_default);
  if (iteratee2 === last_default(mapped)) {
    iteratee2 = void 0;
  } else {
    mapped.pop();
  }
  return mapped.length && mapped[0] === arrays[0] ? baseIntersection_default(mapped, baseIteratee_default(iteratee2, 2)) : [];
});
var intersectionBy_default = intersectionBy;

// node_modules/lodash-es/intersectionWith.js
var intersectionWith = baseRest_default(function(arrays) {
  var comparator = last_default(arrays), mapped = arrayMap_default(arrays, castArrayLikeObject_default);
  comparator = typeof comparator == "function" ? comparator : void 0;
  if (comparator) {
    mapped.pop();
  }
  return mapped.length && mapped[0] === arrays[0] ? baseIntersection_default(mapped, void 0, comparator) : [];
});
var intersectionWith_default = intersectionWith;

// node_modules/lodash-es/_baseInverter.js
function baseInverter(object, setter, iteratee2, accumulator) {
  baseForOwn_default(object, function(value, key, object2) {
    setter(accumulator, iteratee2(value), key, object2);
  });
  return accumulator;
}
var baseInverter_default = baseInverter;

// node_modules/lodash-es/_createInverter.js
function createInverter(setter, toIteratee) {
  return function(object, iteratee2) {
    return baseInverter_default(object, setter, toIteratee(iteratee2), {});
  };
}
var createInverter_default = createInverter;

// node_modules/lodash-es/invert.js
var objectProto24 = Object.prototype;
var nativeObjectToString3 = objectProto24.toString;
var invert = createInverter_default(function(result2, value, key) {
  if (value != null && typeof value.toString != "function") {
    value = nativeObjectToString3.call(value);
  }
  result2[value] = key;
}, constant_default(identity_default));
var invert_default = invert;

// node_modules/lodash-es/invertBy.js
var objectProto25 = Object.prototype;
var hasOwnProperty21 = objectProto25.hasOwnProperty;
var nativeObjectToString4 = objectProto25.toString;
var invertBy = createInverter_default(function(result2, value, key) {
  if (value != null && typeof value.toString != "function") {
    value = nativeObjectToString4.call(value);
  }
  if (hasOwnProperty21.call(result2, value)) {
    result2[value].push(key);
  } else {
    result2[value] = [key];
  }
}, baseIteratee_default);
var invertBy_default = invertBy;

// node_modules/lodash-es/_parent.js
function parent(object, path) {
  return path.length < 2 ? object : baseGet_default(object, baseSlice_default(path, 0, -1));
}
var parent_default = parent;

// node_modules/lodash-es/_baseInvoke.js
function baseInvoke(object, path, args) {
  path = castPath_default(path, object);
  object = parent_default(object, path);
  var func = object == null ? object : object[toKey_default(last_default(path))];
  return func == null ? void 0 : apply_default(func, object, args);
}
var baseInvoke_default = baseInvoke;

// node_modules/lodash-es/invoke.js
var invoke = baseRest_default(baseInvoke_default);
var invoke_default = invoke;

// node_modules/lodash-es/invokeMap.js
var invokeMap = baseRest_default(function(collection, path, args) {
  var index = -1, isFunc = typeof path == "function", result2 = isArrayLike_default(collection) ? Array(collection.length) : [];
  baseEach_default(collection, function(value) {
    result2[++index] = isFunc ? apply_default(path, value, args) : baseInvoke_default(value, path, args);
  });
  return result2;
});
var invokeMap_default = invokeMap;

// node_modules/lodash-es/_baseIsArrayBuffer.js
var arrayBufferTag5 = "[object ArrayBuffer]";
function baseIsArrayBuffer(value) {
  return isObjectLike_default(value) && baseGetTag_default(value) == arrayBufferTag5;
}
var baseIsArrayBuffer_default = baseIsArrayBuffer;

// node_modules/lodash-es/isArrayBuffer.js
var nodeIsArrayBuffer = nodeUtil_default && nodeUtil_default.isArrayBuffer;
var isArrayBuffer = nodeIsArrayBuffer ? baseUnary_default(nodeIsArrayBuffer) : baseIsArrayBuffer_default;
var isArrayBuffer_default = isArrayBuffer;

// node_modules/lodash-es/isBoolean.js
var boolTag5 = "[object Boolean]";
function isBoolean(value) {
  return value === true || value === false || isObjectLike_default(value) && baseGetTag_default(value) == boolTag5;
}
var isBoolean_default = isBoolean;

// node_modules/lodash-es/_baseIsDate.js
var dateTag5 = "[object Date]";
function baseIsDate(value) {
  return isObjectLike_default(value) && baseGetTag_default(value) == dateTag5;
}
var baseIsDate_default = baseIsDate;

// node_modules/lodash-es/isDate.js
var nodeIsDate = nodeUtil_default && nodeUtil_default.isDate;
var isDate = nodeIsDate ? baseUnary_default(nodeIsDate) : baseIsDate_default;
var isDate_default = isDate;

// node_modules/lodash-es/isElement.js
function isElement(value) {
  return isObjectLike_default(value) && value.nodeType === 1 && !isPlainObject_default(value);
}
var isElement_default = isElement;

// node_modules/lodash-es/isEmpty.js
var mapTag8 = "[object Map]";
var setTag8 = "[object Set]";
var objectProto26 = Object.prototype;
var hasOwnProperty22 = objectProto26.hasOwnProperty;
function isEmpty(value) {
  if (value == null) {
    return true;
  }
  if (isArrayLike_default(value) && (isArray_default(value) || typeof value == "string" || typeof value.splice == "function" || isBuffer_default(value) || isTypedArray_default(value) || isArguments_default(value))) {
    return !value.length;
  }
  var tag = getTag_default(value);
  if (tag == mapTag8 || tag == setTag8) {
    return !value.size;
  }
  if (isPrototype_default(value)) {
    return !baseKeys_default(value).length;
  }
  for (var key in value) {
    if (hasOwnProperty22.call(value, key)) {
      return false;
    }
  }
  return true;
}
var isEmpty_default = isEmpty;

// node_modules/lodash-es/isEqual.js
function isEqual(value, other) {
  return baseIsEqual_default(value, other);
}
var isEqual_default = isEqual;

// node_modules/lodash-es/isEqualWith.js
function isEqualWith(value, other, customizer) {
  customizer = typeof customizer == "function" ? customizer : void 0;
  var result2 = customizer ? customizer(value, other) : void 0;
  return result2 === void 0 ? baseIsEqual_default(value, other, void 0, customizer) : !!result2;
}
var isEqualWith_default = isEqualWith;

// node_modules/lodash-es/isFinite.js
var nativeIsFinite2 = root_default.isFinite;
function isFinite(value) {
  return typeof value == "number" && nativeIsFinite2(value);
}
var isFinite_default = isFinite;

// node_modules/lodash-es/isInteger.js
function isInteger(value) {
  return typeof value == "number" && value == toInteger_default(value);
}
var isInteger_default = isInteger;

// node_modules/lodash-es/isMatch.js
function isMatch(object, source) {
  return object === source || baseIsMatch_default(object, source, getMatchData_default(source));
}
var isMatch_default = isMatch;

// node_modules/lodash-es/isMatchWith.js
function isMatchWith(object, source, customizer) {
  customizer = typeof customizer == "function" ? customizer : void 0;
  return baseIsMatch_default(object, source, getMatchData_default(source), customizer);
}
var isMatchWith_default = isMatchWith;

// node_modules/lodash-es/isNumber.js
var numberTag5 = "[object Number]";
function isNumber(value) {
  return typeof value == "number" || isObjectLike_default(value) && baseGetTag_default(value) == numberTag5;
}
var isNumber_default = isNumber;

// node_modules/lodash-es/isNaN.js
function isNaN2(value) {
  return isNumber_default(value) && value != +value;
}
var isNaN_default = isNaN2;

// node_modules/lodash-es/_isMaskable.js
var isMaskable = coreJsData_default ? isFunction_default : stubFalse_default;
var isMaskable_default = isMaskable;

// node_modules/lodash-es/isNative.js
var CORE_ERROR_TEXT = "Unsupported core-js use. Try https://npms.io/search?q=ponyfill.";
function isNative(value) {
  if (isMaskable_default(value)) {
    throw new Error(CORE_ERROR_TEXT);
  }
  return baseIsNative_default(value);
}
var isNative_default = isNative;

// node_modules/lodash-es/isNil.js
function isNil(value) {
  return value == null;
}
var isNil_default = isNil;

// node_modules/lodash-es/isNull.js
function isNull(value) {
  return value === null;
}
var isNull_default = isNull;

// node_modules/lodash-es/_baseIsRegExp.js
var regexpTag5 = "[object RegExp]";
function baseIsRegExp(value) {
  return isObjectLike_default(value) && baseGetTag_default(value) == regexpTag5;
}
var baseIsRegExp_default = baseIsRegExp;

// node_modules/lodash-es/isRegExp.js
var nodeIsRegExp = nodeUtil_default && nodeUtil_default.isRegExp;
var isRegExp = nodeIsRegExp ? baseUnary_default(nodeIsRegExp) : baseIsRegExp_default;
var isRegExp_default = isRegExp;

// node_modules/lodash-es/isSafeInteger.js
var MAX_SAFE_INTEGER3 = 9007199254740991;
function isSafeInteger(value) {
  return isInteger_default(value) && value >= -MAX_SAFE_INTEGER3 && value <= MAX_SAFE_INTEGER3;
}
var isSafeInteger_default = isSafeInteger;

// node_modules/lodash-es/isUndefined.js
function isUndefined(value) {
  return value === void 0;
}
var isUndefined_default = isUndefined;

// node_modules/lodash-es/isWeakMap.js
var weakMapTag4 = "[object WeakMap]";
function isWeakMap(value) {
  return isObjectLike_default(value) && getTag_default(value) == weakMapTag4;
}
var isWeakMap_default = isWeakMap;

// node_modules/lodash-es/isWeakSet.js
var weakSetTag = "[object WeakSet]";
function isWeakSet(value) {
  return isObjectLike_default(value) && baseGetTag_default(value) == weakSetTag;
}
var isWeakSet_default = isWeakSet;

// node_modules/lodash-es/iteratee.js
var CLONE_DEEP_FLAG5 = 1;
function iteratee(func) {
  return baseIteratee_default(typeof func == "function" ? func : baseClone_default(func, CLONE_DEEP_FLAG5));
}
var iteratee_default = iteratee;

// node_modules/lodash-es/join.js
var arrayProto2 = Array.prototype;
var nativeJoin = arrayProto2.join;
function join(array, separator) {
  return array == null ? "" : nativeJoin.call(array, separator);
}
var join_default = join;

// node_modules/lodash-es/kebabCase.js
var kebabCase = createCompounder_default(function(result2, word, index) {
  return result2 + (index ? "-" : "") + word.toLowerCase();
});
var kebabCase_default = kebabCase;

// node_modules/lodash-es/keyBy.js
var keyBy = createAggregator_default(function(result2, value, key) {
  baseAssignValue_default(result2, key, value);
});
var keyBy_default = keyBy;

// node_modules/lodash-es/_strictLastIndexOf.js
function strictLastIndexOf(array, value, fromIndex) {
  var index = fromIndex + 1;
  while (index--) {
    if (array[index] === value) {
      return index;
    }
  }
  return index;
}
var strictLastIndexOf_default = strictLastIndexOf;

// node_modules/lodash-es/lastIndexOf.js
var nativeMax12 = Math.max;
var nativeMin8 = Math.min;
function lastIndexOf(array, value, fromIndex) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return -1;
  }
  var index = length;
  if (fromIndex !== void 0) {
    index = toInteger_default(fromIndex);
    index = index < 0 ? nativeMax12(length + index, 0) : nativeMin8(index, length - 1);
  }
  return value === value ? strictLastIndexOf_default(array, value, index) : baseFindIndex_default(array, baseIsNaN_default, index, true);
}
var lastIndexOf_default = lastIndexOf;

// node_modules/lodash-es/lowerCase.js
var lowerCase = createCompounder_default(function(result2, word, index) {
  return result2 + (index ? " " : "") + word.toLowerCase();
});
var lowerCase_default = lowerCase;

// node_modules/lodash-es/lowerFirst.js
var lowerFirst = createCaseFirst_default("toLowerCase");
var lowerFirst_default = lowerFirst;

// node_modules/lodash-es/_baseLt.js
function baseLt(value, other) {
  return value < other;
}
var baseLt_default = baseLt;

// node_modules/lodash-es/lt.js
var lt = createRelationalOperation_default(baseLt_default);
var lt_default = lt;

// node_modules/lodash-es/lte.js
var lte = createRelationalOperation_default(function(value, other) {
  return value <= other;
});
var lte_default = lte;

// node_modules/lodash-es/mapKeys.js
function mapKeys(object, iteratee2) {
  var result2 = {};
  iteratee2 = baseIteratee_default(iteratee2, 3);
  baseForOwn_default(object, function(value, key, object2) {
    baseAssignValue_default(result2, iteratee2(value, key, object2), value);
  });
  return result2;
}
var mapKeys_default = mapKeys;

// node_modules/lodash-es/mapValues.js
function mapValues(object, iteratee2) {
  var result2 = {};
  iteratee2 = baseIteratee_default(iteratee2, 3);
  baseForOwn_default(object, function(value, key, object2) {
    baseAssignValue_default(result2, key, iteratee2(value, key, object2));
  });
  return result2;
}
var mapValues_default = mapValues;

// node_modules/lodash-es/matches.js
var CLONE_DEEP_FLAG6 = 1;
function matches(source) {
  return baseMatches_default(baseClone_default(source, CLONE_DEEP_FLAG6));
}
var matches_default = matches;

// node_modules/lodash-es/matchesProperty.js
var CLONE_DEEP_FLAG7 = 1;
function matchesProperty(path, srcValue) {
  return baseMatchesProperty_default(path, baseClone_default(srcValue, CLONE_DEEP_FLAG7));
}
var matchesProperty_default = matchesProperty;

// node_modules/lodash-es/_baseExtremum.js
function baseExtremum(array, iteratee2, comparator) {
  var index = -1, length = array.length;
  while (++index < length) {
    var value = array[index], current = iteratee2(value);
    if (current != null && (computed === void 0 ? current === current && !isSymbol_default(current) : comparator(current, computed))) {
      var computed = current, result2 = value;
    }
  }
  return result2;
}
var baseExtremum_default = baseExtremum;

// node_modules/lodash-es/max.js
function max(array) {
  return array && array.length ? baseExtremum_default(array, identity_default, baseGt_default) : void 0;
}
var max_default = max;

// node_modules/lodash-es/maxBy.js
function maxBy(array, iteratee2) {
  return array && array.length ? baseExtremum_default(array, baseIteratee_default(iteratee2, 2), baseGt_default) : void 0;
}
var maxBy_default = maxBy;

// node_modules/lodash-es/_baseSum.js
function baseSum(array, iteratee2) {
  var result2, index = -1, length = array.length;
  while (++index < length) {
    var current = iteratee2(array[index]);
    if (current !== void 0) {
      result2 = result2 === void 0 ? current : result2 + current;
    }
  }
  return result2;
}
var baseSum_default = baseSum;

// node_modules/lodash-es/_baseMean.js
var NAN3 = 0 / 0;
function baseMean(array, iteratee2) {
  var length = array == null ? 0 : array.length;
  return length ? baseSum_default(array, iteratee2) / length : NAN3;
}
var baseMean_default = baseMean;

// node_modules/lodash-es/mean.js
function mean(array) {
  return baseMean_default(array, identity_default);
}
var mean_default = mean;

// node_modules/lodash-es/meanBy.js
function meanBy(array, iteratee2) {
  return baseMean_default(array, baseIteratee_default(iteratee2, 2));
}
var meanBy_default = meanBy;

// node_modules/lodash-es/merge.js
var merge = createAssigner_default(function(object, source, srcIndex) {
  baseMerge_default(object, source, srcIndex);
});
var merge_default = merge;

// node_modules/lodash-es/method.js
var method = baseRest_default(function(path, args) {
  return function(object) {
    return baseInvoke_default(object, path, args);
  };
});
var method_default = method;

// node_modules/lodash-es/methodOf.js
var methodOf = baseRest_default(function(object, args) {
  return function(path) {
    return baseInvoke_default(object, path, args);
  };
});
var methodOf_default = methodOf;

// node_modules/lodash-es/min.js
function min(array) {
  return array && array.length ? baseExtremum_default(array, identity_default, baseLt_default) : void 0;
}
var min_default = min;

// node_modules/lodash-es/minBy.js
function minBy(array, iteratee2) {
  return array && array.length ? baseExtremum_default(array, baseIteratee_default(iteratee2, 2), baseLt_default) : void 0;
}
var minBy_default = minBy;

// node_modules/lodash-es/mixin.js
function mixin(object, source, options) {
  var props = keys_default(source), methodNames = baseFunctions_default(source, props);
  var chain2 = !(isObject_default(options) && "chain" in options) || !!options.chain, isFunc = isFunction_default(object);
  arrayEach_default(methodNames, function(methodName) {
    var func = source[methodName];
    object[methodName] = func;
    if (isFunc) {
      object.prototype[methodName] = function() {
        var chainAll = this.__chain__;
        if (chain2 || chainAll) {
          var result2 = object(this.__wrapped__), actions = result2.__actions__ = copyArray_default(this.__actions__);
          actions.push({ "func": func, "args": arguments, "thisArg": object });
          result2.__chain__ = chainAll;
          return result2;
        }
        return func.apply(object, arrayPush_default([this.value()], arguments));
      };
    }
  });
  return object;
}
var mixin_default = mixin;

// node_modules/lodash-es/multiply.js
var multiply = createMathOperation_default(function(multiplier, multiplicand) {
  return multiplier * multiplicand;
}, 1);
var multiply_default = multiply;

// node_modules/lodash-es/negate.js
var FUNC_ERROR_TEXT9 = "Expected a function";
function negate(predicate) {
  if (typeof predicate != "function") {
    throw new TypeError(FUNC_ERROR_TEXT9);
  }
  return function() {
    var args = arguments;
    switch (args.length) {
      case 0:
        return !predicate.call(this);
      case 1:
        return !predicate.call(this, args[0]);
      case 2:
        return !predicate.call(this, args[0], args[1]);
      case 3:
        return !predicate.call(this, args[0], args[1], args[2]);
    }
    return !predicate.apply(this, args);
  };
}
var negate_default = negate;

// node_modules/lodash-es/_iteratorToArray.js
function iteratorToArray(iterator) {
  var data, result2 = [];
  while (!(data = iterator.next()).done) {
    result2.push(data.value);
  }
  return result2;
}
var iteratorToArray_default = iteratorToArray;

// node_modules/lodash-es/toArray.js
var mapTag9 = "[object Map]";
var setTag9 = "[object Set]";
var symIterator = Symbol_default ? Symbol_default.iterator : void 0;
function toArray(value) {
  if (!value) {
    return [];
  }
  if (isArrayLike_default(value)) {
    return isString_default(value) ? stringToArray_default(value) : copyArray_default(value);
  }
  if (symIterator && value[symIterator]) {
    return iteratorToArray_default(value[symIterator]());
  }
  var tag = getTag_default(value), func = tag == mapTag9 ? mapToArray_default : tag == setTag9 ? setToArray_default : values_default;
  return func(value);
}
var toArray_default = toArray;

// node_modules/lodash-es/next.js
function wrapperNext() {
  if (this.__values__ === void 0) {
    this.__values__ = toArray_default(this.value());
  }
  var done = this.__index__ >= this.__values__.length, value = done ? void 0 : this.__values__[this.__index__++];
  return { "done": done, "value": value };
}
var next_default = wrapperNext;

// node_modules/lodash-es/_baseNth.js
function baseNth(array, n) {
  var length = array.length;
  if (!length) {
    return;
  }
  n += n < 0 ? length : 0;
  return isIndex_default(n, length) ? array[n] : void 0;
}
var baseNth_default = baseNth;

// node_modules/lodash-es/nth.js
function nth(array, n) {
  return array && array.length ? baseNth_default(array, toInteger_default(n)) : void 0;
}
var nth_default = nth;

// node_modules/lodash-es/nthArg.js
function nthArg(n) {
  n = toInteger_default(n);
  return baseRest_default(function(args) {
    return baseNth_default(args, n);
  });
}
var nthArg_default = nthArg;

// node_modules/lodash-es/_baseUnset.js
function baseUnset(object, path) {
  path = castPath_default(path, object);
  object = parent_default(object, path);
  return object == null || delete object[toKey_default(last_default(path))];
}
var baseUnset_default = baseUnset;

// node_modules/lodash-es/_customOmitClone.js
function customOmitClone(value) {
  return isPlainObject_default(value) ? void 0 : value;
}
var customOmitClone_default = customOmitClone;

// node_modules/lodash-es/omit.js
var CLONE_DEEP_FLAG8 = 1;
var CLONE_FLAT_FLAG2 = 2;
var CLONE_SYMBOLS_FLAG6 = 4;
var omit = flatRest_default(function(object, paths) {
  var result2 = {};
  if (object == null) {
    return result2;
  }
  var isDeep = false;
  paths = arrayMap_default(paths, function(path) {
    path = castPath_default(path, object);
    isDeep || (isDeep = path.length > 1);
    return path;
  });
  copyObject_default(object, getAllKeysIn_default(object), result2);
  if (isDeep) {
    result2 = baseClone_default(result2, CLONE_DEEP_FLAG8 | CLONE_FLAT_FLAG2 | CLONE_SYMBOLS_FLAG6, customOmitClone_default);
  }
  var length = paths.length;
  while (length--) {
    baseUnset_default(result2, paths[length]);
  }
  return result2;
});
var omit_default = omit;

// node_modules/lodash-es/_baseSet.js
function baseSet(object, path, value, customizer) {
  if (!isObject_default(object)) {
    return object;
  }
  path = castPath_default(path, object);
  var index = -1, length = path.length, lastIndex = length - 1, nested = object;
  while (nested != null && ++index < length) {
    var key = toKey_default(path[index]), newValue = value;
    if (key === "__proto__" || key === "constructor" || key === "prototype") {
      return object;
    }
    if (index != lastIndex) {
      var objValue = nested[key];
      newValue = customizer ? customizer(objValue, key, nested) : void 0;
      if (newValue === void 0) {
        newValue = isObject_default(objValue) ? objValue : isIndex_default(path[index + 1]) ? [] : {};
      }
    }
    assignValue_default(nested, key, newValue);
    nested = nested[key];
  }
  return object;
}
var baseSet_default = baseSet;

// node_modules/lodash-es/_basePickBy.js
function basePickBy(object, paths, predicate) {
  var index = -1, length = paths.length, result2 = {};
  while (++index < length) {
    var path = paths[index], value = baseGet_default(object, path);
    if (predicate(value, path)) {
      baseSet_default(result2, castPath_default(path, object), value);
    }
  }
  return result2;
}
var basePickBy_default = basePickBy;

// node_modules/lodash-es/pickBy.js
function pickBy(object, predicate) {
  if (object == null) {
    return {};
  }
  var props = arrayMap_default(getAllKeysIn_default(object), function(prop) {
    return [prop];
  });
  predicate = baseIteratee_default(predicate);
  return basePickBy_default(object, props, function(value, path) {
    return predicate(value, path[0]);
  });
}
var pickBy_default = pickBy;

// node_modules/lodash-es/omitBy.js
function omitBy(object, predicate) {
  return pickBy_default(object, negate_default(baseIteratee_default(predicate)));
}
var omitBy_default = omitBy;

// node_modules/lodash-es/once.js
function once(func) {
  return before_default(2, func);
}
var once_default = once;

// node_modules/lodash-es/_baseSortBy.js
function baseSortBy(array, comparer) {
  var length = array.length;
  array.sort(comparer);
  while (length--) {
    array[length] = array[length].value;
  }
  return array;
}
var baseSortBy_default = baseSortBy;

// node_modules/lodash-es/_compareAscending.js
function compareAscending(value, other) {
  if (value !== other) {
    var valIsDefined = value !== void 0, valIsNull = value === null, valIsReflexive = value === value, valIsSymbol = isSymbol_default(value);
    var othIsDefined = other !== void 0, othIsNull = other === null, othIsReflexive = other === other, othIsSymbol = isSymbol_default(other);
    if (!othIsNull && !othIsSymbol && !valIsSymbol && value > other || valIsSymbol && othIsDefined && othIsReflexive && !othIsNull && !othIsSymbol || valIsNull && othIsDefined && othIsReflexive || !valIsDefined && othIsReflexive || !valIsReflexive) {
      return 1;
    }
    if (!valIsNull && !valIsSymbol && !othIsSymbol && value < other || othIsSymbol && valIsDefined && valIsReflexive && !valIsNull && !valIsSymbol || othIsNull && valIsDefined && valIsReflexive || !othIsDefined && valIsReflexive || !othIsReflexive) {
      return -1;
    }
  }
  return 0;
}
var compareAscending_default = compareAscending;

// node_modules/lodash-es/_compareMultiple.js
function compareMultiple(object, other, orders) {
  var index = -1, objCriteria = object.criteria, othCriteria = other.criteria, length = objCriteria.length, ordersLength = orders.length;
  while (++index < length) {
    var result2 = compareAscending_default(objCriteria[index], othCriteria[index]);
    if (result2) {
      if (index >= ordersLength) {
        return result2;
      }
      var order = orders[index];
      return result2 * (order == "desc" ? -1 : 1);
    }
  }
  return object.index - other.index;
}
var compareMultiple_default = compareMultiple;

// node_modules/lodash-es/_baseOrderBy.js
function baseOrderBy(collection, iteratees, orders) {
  if (iteratees.length) {
    iteratees = arrayMap_default(iteratees, function(iteratee2) {
      if (isArray_default(iteratee2)) {
        return function(value) {
          return baseGet_default(value, iteratee2.length === 1 ? iteratee2[0] : iteratee2);
        };
      }
      return iteratee2;
    });
  } else {
    iteratees = [identity_default];
  }
  var index = -1;
  iteratees = arrayMap_default(iteratees, baseUnary_default(baseIteratee_default));
  var result2 = baseMap_default(collection, function(value, key, collection2) {
    var criteria = arrayMap_default(iteratees, function(iteratee2) {
      return iteratee2(value);
    });
    return { "criteria": criteria, "index": ++index, "value": value };
  });
  return baseSortBy_default(result2, function(object, other) {
    return compareMultiple_default(object, other, orders);
  });
}
var baseOrderBy_default = baseOrderBy;

// node_modules/lodash-es/orderBy.js
function orderBy(collection, iteratees, orders, guard) {
  if (collection == null) {
    return [];
  }
  if (!isArray_default(iteratees)) {
    iteratees = iteratees == null ? [] : [iteratees];
  }
  orders = guard ? void 0 : orders;
  if (!isArray_default(orders)) {
    orders = orders == null ? [] : [orders];
  }
  return baseOrderBy_default(collection, iteratees, orders);
}
var orderBy_default = orderBy;

// node_modules/lodash-es/_createOver.js
function createOver(arrayFunc) {
  return flatRest_default(function(iteratees) {
    iteratees = arrayMap_default(iteratees, baseUnary_default(baseIteratee_default));
    return baseRest_default(function(args) {
      var thisArg = this;
      return arrayFunc(iteratees, function(iteratee2) {
        return apply_default(iteratee2, thisArg, args);
      });
    });
  });
}
var createOver_default = createOver;

// node_modules/lodash-es/over.js
var over = createOver_default(arrayMap_default);
var over_default = over;

// node_modules/lodash-es/_castRest.js
var castRest = baseRest_default;
var castRest_default = castRest;

// node_modules/lodash-es/overArgs.js
var nativeMin9 = Math.min;
var overArgs = castRest_default(function(func, transforms) {
  transforms = transforms.length == 1 && isArray_default(transforms[0]) ? arrayMap_default(transforms[0], baseUnary_default(baseIteratee_default)) : arrayMap_default(baseFlatten_default(transforms, 1), baseUnary_default(baseIteratee_default));
  var funcsLength = transforms.length;
  return baseRest_default(function(args) {
    var index = -1, length = nativeMin9(args.length, funcsLength);
    while (++index < length) {
      args[index] = transforms[index].call(this, args[index]);
    }
    return apply_default(func, this, args);
  });
});
var overArgs_default = overArgs;

// node_modules/lodash-es/overEvery.js
var overEvery = createOver_default(arrayEvery_default);
var overEvery_default = overEvery;

// node_modules/lodash-es/overSome.js
var overSome = createOver_default(arraySome_default);
var overSome_default = overSome;

// node_modules/lodash-es/_baseRepeat.js
var MAX_SAFE_INTEGER4 = 9007199254740991;
var nativeFloor = Math.floor;
function baseRepeat(string, n) {
  var result2 = "";
  if (!string || n < 1 || n > MAX_SAFE_INTEGER4) {
    return result2;
  }
  do {
    if (n % 2) {
      result2 += string;
    }
    n = nativeFloor(n / 2);
    if (n) {
      string += string;
    }
  } while (n);
  return result2;
}
var baseRepeat_default = baseRepeat;

// node_modules/lodash-es/_asciiSize.js
var asciiSize = baseProperty_default("length");
var asciiSize_default = asciiSize;

// node_modules/lodash-es/_unicodeSize.js
var rsAstralRange4 = "\\ud800-\\udfff";
var rsComboMarksRange5 = "\\u0300-\\u036f";
var reComboHalfMarksRange5 = "\\ufe20-\\ufe2f";
var rsComboSymbolsRange5 = "\\u20d0-\\u20ff";
var rsComboRange5 = rsComboMarksRange5 + reComboHalfMarksRange5 + rsComboSymbolsRange5;
var rsVarRange4 = "\\ufe0e\\ufe0f";
var rsAstral2 = "[" + rsAstralRange4 + "]";
var rsCombo4 = "[" + rsComboRange5 + "]";
var rsFitz3 = "\\ud83c[\\udffb-\\udfff]";
var rsModifier3 = "(?:" + rsCombo4 + "|" + rsFitz3 + ")";
var rsNonAstral3 = "[^" + rsAstralRange4 + "]";
var rsRegional3 = "(?:\\ud83c[\\udde6-\\uddff]){2}";
var rsSurrPair3 = "[\\ud800-\\udbff][\\udc00-\\udfff]";
var rsZWJ4 = "\\u200d";
var reOptMod3 = rsModifier3 + "?";
var rsOptVar3 = "[" + rsVarRange4 + "]?";
var rsOptJoin3 = "(?:" + rsZWJ4 + "(?:" + [rsNonAstral3, rsRegional3, rsSurrPair3].join("|") + ")" + rsOptVar3 + reOptMod3 + ")*";
var rsSeq3 = rsOptVar3 + reOptMod3 + rsOptJoin3;
var rsSymbol2 = "(?:" + [rsNonAstral3 + rsCombo4 + "?", rsCombo4, rsRegional3, rsSurrPair3, rsAstral2].join("|") + ")";
var reUnicode2 = RegExp(rsFitz3 + "(?=" + rsFitz3 + ")|" + rsSymbol2 + rsSeq3, "g");
function unicodeSize(string) {
  var result2 = reUnicode2.lastIndex = 0;
  while (reUnicode2.test(string)) {
    ++result2;
  }
  return result2;
}
var unicodeSize_default = unicodeSize;

// node_modules/lodash-es/_stringSize.js
function stringSize(string) {
  return hasUnicode_default(string) ? unicodeSize_default(string) : asciiSize_default(string);
}
var stringSize_default = stringSize;

// node_modules/lodash-es/_createPadding.js
var nativeCeil2 = Math.ceil;
function createPadding(length, chars) {
  chars = chars === void 0 ? " " : baseToString_default(chars);
  var charsLength = chars.length;
  if (charsLength < 2) {
    return charsLength ? baseRepeat_default(chars, length) : chars;
  }
  var result2 = baseRepeat_default(chars, nativeCeil2(length / stringSize_default(chars)));
  return hasUnicode_default(chars) ? castSlice_default(stringToArray_default(result2), 0, length).join("") : result2.slice(0, length);
}
var createPadding_default = createPadding;

// node_modules/lodash-es/pad.js
var nativeCeil3 = Math.ceil;
var nativeFloor2 = Math.floor;
function pad(string, length, chars) {
  string = toString_default(string);
  length = toInteger_default(length);
  var strLength = length ? stringSize_default(string) : 0;
  if (!length || strLength >= length) {
    return string;
  }
  var mid = (length - strLength) / 2;
  return createPadding_default(nativeFloor2(mid), chars) + string + createPadding_default(nativeCeil3(mid), chars);
}
var pad_default = pad;

// node_modules/lodash-es/padEnd.js
function padEnd(string, length, chars) {
  string = toString_default(string);
  length = toInteger_default(length);
  var strLength = length ? stringSize_default(string) : 0;
  return length && strLength < length ? string + createPadding_default(length - strLength, chars) : string;
}
var padEnd_default = padEnd;

// node_modules/lodash-es/padStart.js
function padStart(string, length, chars) {
  string = toString_default(string);
  length = toInteger_default(length);
  var strLength = length ? stringSize_default(string) : 0;
  return length && strLength < length ? createPadding_default(length - strLength, chars) + string : string;
}
var padStart_default = padStart;

// node_modules/lodash-es/parseInt.js
var reTrimStart2 = /^\s+/;
var nativeParseInt = root_default.parseInt;
function parseInt2(string, radix, guard) {
  if (guard || radix == null) {
    radix = 0;
  } else if (radix) {
    radix = +radix;
  }
  return nativeParseInt(toString_default(string).replace(reTrimStart2, ""), radix || 0);
}
var parseInt_default = parseInt2;

// node_modules/lodash-es/partial.js
var WRAP_PARTIAL_FLAG7 = 32;
var partial = baseRest_default(function(func, partials) {
  var holders = replaceHolders_default(partials, getHolder_default(partial));
  return createWrap_default(func, WRAP_PARTIAL_FLAG7, void 0, partials, holders);
});
partial.placeholder = {};
var partial_default = partial;

// node_modules/lodash-es/partialRight.js
var WRAP_PARTIAL_RIGHT_FLAG4 = 64;
var partialRight = baseRest_default(function(func, partials) {
  var holders = replaceHolders_default(partials, getHolder_default(partialRight));
  return createWrap_default(func, WRAP_PARTIAL_RIGHT_FLAG4, void 0, partials, holders);
});
partialRight.placeholder = {};
var partialRight_default = partialRight;

// node_modules/lodash-es/partition.js
var partition = createAggregator_default(function(result2, value, key) {
  result2[key ? 0 : 1].push(value);
}, function() {
  return [[], []];
});
var partition_default = partition;

// node_modules/lodash-es/_basePick.js
function basePick(object, paths) {
  return basePickBy_default(object, paths, function(value, path) {
    return hasIn_default(object, path);
  });
}
var basePick_default = basePick;

// node_modules/lodash-es/pick.js
var pick = flatRest_default(function(object, paths) {
  return object == null ? {} : basePick_default(object, paths);
});
var pick_default = pick;

// node_modules/lodash-es/plant.js
function wrapperPlant(value) {
  var result2, parent2 = this;
  while (parent2 instanceof baseLodash_default) {
    var clone2 = wrapperClone_default(parent2);
    clone2.__index__ = 0;
    clone2.__values__ = void 0;
    if (result2) {
      previous.__wrapped__ = clone2;
    } else {
      result2 = clone2;
    }
    var previous = clone2;
    parent2 = parent2.__wrapped__;
  }
  previous.__wrapped__ = value;
  return result2;
}
var plant_default = wrapperPlant;

// node_modules/lodash-es/propertyOf.js
function propertyOf(object) {
  return function(path) {
    return object == null ? void 0 : baseGet_default(object, path);
  };
}
var propertyOf_default = propertyOf;

// node_modules/lodash-es/_baseIndexOfWith.js
function baseIndexOfWith(array, value, fromIndex, comparator) {
  var index = fromIndex - 1, length = array.length;
  while (++index < length) {
    if (comparator(array[index], value)) {
      return index;
    }
  }
  return -1;
}
var baseIndexOfWith_default = baseIndexOfWith;

// node_modules/lodash-es/_basePullAll.js
var arrayProto3 = Array.prototype;
var splice2 = arrayProto3.splice;
function basePullAll(array, values2, iteratee2, comparator) {
  var indexOf2 = comparator ? baseIndexOfWith_default : baseIndexOf_default, index = -1, length = values2.length, seen = array;
  if (array === values2) {
    values2 = copyArray_default(values2);
  }
  if (iteratee2) {
    seen = arrayMap_default(array, baseUnary_default(iteratee2));
  }
  while (++index < length) {
    var fromIndex = 0, value = values2[index], computed = iteratee2 ? iteratee2(value) : value;
    while ((fromIndex = indexOf2(seen, computed, fromIndex, comparator)) > -1) {
      if (seen !== array) {
        splice2.call(seen, fromIndex, 1);
      }
      splice2.call(array, fromIndex, 1);
    }
  }
  return array;
}
var basePullAll_default = basePullAll;

// node_modules/lodash-es/pullAll.js
function pullAll(array, values2) {
  return array && array.length && values2 && values2.length ? basePullAll_default(array, values2) : array;
}
var pullAll_default = pullAll;

// node_modules/lodash-es/pull.js
var pull = baseRest_default(pullAll_default);
var pull_default = pull;

// node_modules/lodash-es/pullAllBy.js
function pullAllBy(array, values2, iteratee2) {
  return array && array.length && values2 && values2.length ? basePullAll_default(array, values2, baseIteratee_default(iteratee2, 2)) : array;
}
var pullAllBy_default = pullAllBy;

// node_modules/lodash-es/pullAllWith.js
function pullAllWith(array, values2, comparator) {
  return array && array.length && values2 && values2.length ? basePullAll_default(array, values2, void 0, comparator) : array;
}
var pullAllWith_default = pullAllWith;

// node_modules/lodash-es/_basePullAt.js
var arrayProto4 = Array.prototype;
var splice3 = arrayProto4.splice;
function basePullAt(array, indexes) {
  var length = array ? indexes.length : 0, lastIndex = length - 1;
  while (length--) {
    var index = indexes[length];
    if (length == lastIndex || index !== previous) {
      var previous = index;
      if (isIndex_default(index)) {
        splice3.call(array, index, 1);
      } else {
        baseUnset_default(array, index);
      }
    }
  }
  return array;
}
var basePullAt_default = basePullAt;

// node_modules/lodash-es/pullAt.js
var pullAt = flatRest_default(function(array, indexes) {
  var length = array == null ? 0 : array.length, result2 = baseAt_default(array, indexes);
  basePullAt_default(array, arrayMap_default(indexes, function(index) {
    return isIndex_default(index, length) ? +index : index;
  }).sort(compareAscending_default));
  return result2;
});
var pullAt_default = pullAt;

// node_modules/lodash-es/_baseRandom.js
var nativeFloor3 = Math.floor;
var nativeRandom = Math.random;
function baseRandom(lower, upper) {
  return lower + nativeFloor3(nativeRandom() * (upper - lower + 1));
}
var baseRandom_default = baseRandom;

// node_modules/lodash-es/random.js
var freeParseFloat = parseFloat;
var nativeMin10 = Math.min;
var nativeRandom2 = Math.random;
function random(lower, upper, floating) {
  if (floating && typeof floating != "boolean" && isIterateeCall_default(lower, upper, floating)) {
    upper = floating = void 0;
  }
  if (floating === void 0) {
    if (typeof upper == "boolean") {
      floating = upper;
      upper = void 0;
    } else if (typeof lower == "boolean") {
      floating = lower;
      lower = void 0;
    }
  }
  if (lower === void 0 && upper === void 0) {
    lower = 0;
    upper = 1;
  } else {
    lower = toFinite_default(lower);
    if (upper === void 0) {
      upper = lower;
      lower = 0;
    } else {
      upper = toFinite_default(upper);
    }
  }
  if (lower > upper) {
    var temp = lower;
    lower = upper;
    upper = temp;
  }
  if (floating || lower % 1 || upper % 1) {
    var rand = nativeRandom2();
    return nativeMin10(lower + rand * (upper - lower + freeParseFloat("1e-" + ((rand + "").length - 1))), upper);
  }
  return baseRandom_default(lower, upper);
}
var random_default = random;

// node_modules/lodash-es/_baseRange.js
var nativeCeil4 = Math.ceil;
var nativeMax13 = Math.max;
function baseRange(start, end, step, fromRight) {
  var index = -1, length = nativeMax13(nativeCeil4((end - start) / (step || 1)), 0), result2 = Array(length);
  while (length--) {
    result2[fromRight ? length : ++index] = start;
    start += step;
  }
  return result2;
}
var baseRange_default = baseRange;

// node_modules/lodash-es/_createRange.js
function createRange(fromRight) {
  return function(start, end, step) {
    if (step && typeof step != "number" && isIterateeCall_default(start, end, step)) {
      end = step = void 0;
    }
    start = toFinite_default(start);
    if (end === void 0) {
      end = start;
      start = 0;
    } else {
      end = toFinite_default(end);
    }
    step = step === void 0 ? start < end ? 1 : -1 : toFinite_default(step);
    return baseRange_default(start, end, step, fromRight);
  };
}
var createRange_default = createRange;

// node_modules/lodash-es/range.js
var range = createRange_default();
var range_default = range;

// node_modules/lodash-es/rangeRight.js
var rangeRight = createRange_default(true);
var rangeRight_default = rangeRight;

// node_modules/lodash-es/rearg.js
var WRAP_REARG_FLAG4 = 256;
var rearg = flatRest_default(function(func, indexes) {
  return createWrap_default(func, WRAP_REARG_FLAG4, void 0, void 0, void 0, indexes);
});
var rearg_default = rearg;

// node_modules/lodash-es/_baseReduce.js
function baseReduce(collection, iteratee2, accumulator, initAccum, eachFunc) {
  eachFunc(collection, function(value, index, collection2) {
    accumulator = initAccum ? (initAccum = false, value) : iteratee2(accumulator, value, index, collection2);
  });
  return accumulator;
}
var baseReduce_default = baseReduce;

// node_modules/lodash-es/reduce.js
function reduce(collection, iteratee2, accumulator) {
  var func = isArray_default(collection) ? arrayReduce_default : baseReduce_default, initAccum = arguments.length < 3;
  return func(collection, baseIteratee_default(iteratee2, 4), accumulator, initAccum, baseEach_default);
}
var reduce_default = reduce;

// node_modules/lodash-es/_arrayReduceRight.js
function arrayReduceRight(array, iteratee2, accumulator, initAccum) {
  var length = array == null ? 0 : array.length;
  if (initAccum && length) {
    accumulator = array[--length];
  }
  while (length--) {
    accumulator = iteratee2(accumulator, array[length], length, array);
  }
  return accumulator;
}
var arrayReduceRight_default = arrayReduceRight;

// node_modules/lodash-es/reduceRight.js
function reduceRight(collection, iteratee2, accumulator) {
  var func = isArray_default(collection) ? arrayReduceRight_default : baseReduce_default, initAccum = arguments.length < 3;
  return func(collection, baseIteratee_default(iteratee2, 4), accumulator, initAccum, baseEachRight_default);
}
var reduceRight_default = reduceRight;

// node_modules/lodash-es/reject.js
function reject(collection, predicate) {
  var func = isArray_default(collection) ? arrayFilter_default : baseFilter_default;
  return func(collection, negate_default(baseIteratee_default(predicate, 3)));
}
var reject_default = reject;

// node_modules/lodash-es/remove.js
function remove(array, predicate) {
  var result2 = [];
  if (!(array && array.length)) {
    return result2;
  }
  var index = -1, indexes = [], length = array.length;
  predicate = baseIteratee_default(predicate, 3);
  while (++index < length) {
    var value = array[index];
    if (predicate(value, index, array)) {
      result2.push(value);
      indexes.push(index);
    }
  }
  basePullAt_default(array, indexes);
  return result2;
}
var remove_default = remove;

// node_modules/lodash-es/repeat.js
function repeat(string, n, guard) {
  if (guard ? isIterateeCall_default(string, n, guard) : n === void 0) {
    n = 1;
  } else {
    n = toInteger_default(n);
  }
  return baseRepeat_default(toString_default(string), n);
}
var repeat_default = repeat;

// node_modules/lodash-es/replace.js
function replace() {
  var args = arguments, string = toString_default(args[0]);
  return args.length < 3 ? string : string.replace(args[1], args[2]);
}
var replace_default = replace;

// node_modules/lodash-es/rest.js
var FUNC_ERROR_TEXT10 = "Expected a function";
function rest(func, start) {
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT10);
  }
  start = start === void 0 ? start : toInteger_default(start);
  return baseRest_default(func, start);
}
var rest_default = rest;

// node_modules/lodash-es/result.js
function result(object, path, defaultValue) {
  path = castPath_default(path, object);
  var index = -1, length = path.length;
  if (!length) {
    length = 1;
    object = void 0;
  }
  while (++index < length) {
    var value = object == null ? void 0 : object[toKey_default(path[index])];
    if (value === void 0) {
      index = length;
      value = defaultValue;
    }
    object = isFunction_default(value) ? value.call(object) : value;
  }
  return object;
}
var result_default = result;

// node_modules/lodash-es/reverse.js
var arrayProto5 = Array.prototype;
var nativeReverse = arrayProto5.reverse;
function reverse(array) {
  return array == null ? array : nativeReverse.call(array);
}
var reverse_default = reverse;

// node_modules/lodash-es/round.js
var round = createRound_default("round");
var round_default = round;

// node_modules/lodash-es/_arraySample.js
function arraySample(array) {
  var length = array.length;
  return length ? array[baseRandom_default(0, length - 1)] : void 0;
}
var arraySample_default = arraySample;

// node_modules/lodash-es/_baseSample.js
function baseSample(collection) {
  return arraySample_default(values_default(collection));
}
var baseSample_default = baseSample;

// node_modules/lodash-es/sample.js
function sample(collection) {
  var func = isArray_default(collection) ? arraySample_default : baseSample_default;
  return func(collection);
}
var sample_default = sample;

// node_modules/lodash-es/_shuffleSelf.js
function shuffleSelf(array, size2) {
  var index = -1, length = array.length, lastIndex = length - 1;
  size2 = size2 === void 0 ? length : size2;
  while (++index < size2) {
    var rand = baseRandom_default(index, lastIndex), value = array[rand];
    array[rand] = array[index];
    array[index] = value;
  }
  array.length = size2;
  return array;
}
var shuffleSelf_default = shuffleSelf;

// node_modules/lodash-es/_arraySampleSize.js
function arraySampleSize(array, n) {
  return shuffleSelf_default(copyArray_default(array), baseClamp_default(n, 0, array.length));
}
var arraySampleSize_default = arraySampleSize;

// node_modules/lodash-es/_baseSampleSize.js
function baseSampleSize(collection, n) {
  var array = values_default(collection);
  return shuffleSelf_default(array, baseClamp_default(n, 0, array.length));
}
var baseSampleSize_default = baseSampleSize;

// node_modules/lodash-es/sampleSize.js
function sampleSize(collection, n, guard) {
  if (guard ? isIterateeCall_default(collection, n, guard) : n === void 0) {
    n = 1;
  } else {
    n = toInteger_default(n);
  }
  var func = isArray_default(collection) ? arraySampleSize_default : baseSampleSize_default;
  return func(collection, n);
}
var sampleSize_default = sampleSize;

// node_modules/lodash-es/set.js
function set(object, path, value) {
  return object == null ? object : baseSet_default(object, path, value);
}
var set_default = set;

// node_modules/lodash-es/setWith.js
function setWith(object, path, value, customizer) {
  customizer = typeof customizer == "function" ? customizer : void 0;
  return object == null ? object : baseSet_default(object, path, value, customizer);
}
var setWith_default = setWith;

// node_modules/lodash-es/_arrayShuffle.js
function arrayShuffle(array) {
  return shuffleSelf_default(copyArray_default(array));
}
var arrayShuffle_default = arrayShuffle;

// node_modules/lodash-es/_baseShuffle.js
function baseShuffle(collection) {
  return shuffleSelf_default(values_default(collection));
}
var baseShuffle_default = baseShuffle;

// node_modules/lodash-es/shuffle.js
function shuffle(collection) {
  var func = isArray_default(collection) ? arrayShuffle_default : baseShuffle_default;
  return func(collection);
}
var shuffle_default = shuffle;

// node_modules/lodash-es/size.js
var mapTag10 = "[object Map]";
var setTag10 = "[object Set]";
function size(collection) {
  if (collection == null) {
    return 0;
  }
  if (isArrayLike_default(collection)) {
    return isString_default(collection) ? stringSize_default(collection) : collection.length;
  }
  var tag = getTag_default(collection);
  if (tag == mapTag10 || tag == setTag10) {
    return collection.size;
  }
  return baseKeys_default(collection).length;
}
var size_default = size;

// node_modules/lodash-es/slice.js
function slice(array, start, end) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return [];
  }
  if (end && typeof end != "number" && isIterateeCall_default(array, start, end)) {
    start = 0;
    end = length;
  } else {
    start = start == null ? 0 : toInteger_default(start);
    end = end === void 0 ? length : toInteger_default(end);
  }
  return baseSlice_default(array, start, end);
}
var slice_default = slice;

// node_modules/lodash-es/snakeCase.js
var snakeCase = createCompounder_default(function(result2, word, index) {
  return result2 + (index ? "_" : "") + word.toLowerCase();
});
var snakeCase_default = snakeCase;

// node_modules/lodash-es/_baseSome.js
function baseSome(collection, predicate) {
  var result2;
  baseEach_default(collection, function(value, index, collection2) {
    result2 = predicate(value, index, collection2);
    return !result2;
  });
  return !!result2;
}
var baseSome_default = baseSome;

// node_modules/lodash-es/some.js
function some(collection, predicate, guard) {
  var func = isArray_default(collection) ? arraySome_default : baseSome_default;
  if (guard && isIterateeCall_default(collection, predicate, guard)) {
    predicate = void 0;
  }
  return func(collection, baseIteratee_default(predicate, 3));
}
var some_default = some;

// node_modules/lodash-es/sortBy.js
var sortBy = baseRest_default(function(collection, iteratees) {
  if (collection == null) {
    return [];
  }
  var length = iteratees.length;
  if (length > 1 && isIterateeCall_default(collection, iteratees[0], iteratees[1])) {
    iteratees = [];
  } else if (length > 2 && isIterateeCall_default(iteratees[0], iteratees[1], iteratees[2])) {
    iteratees = [iteratees[0]];
  }
  return baseOrderBy_default(collection, baseFlatten_default(iteratees, 1), []);
});
var sortBy_default = sortBy;

// node_modules/lodash-es/_baseSortedIndexBy.js
var MAX_ARRAY_LENGTH3 = 4294967295;
var MAX_ARRAY_INDEX = MAX_ARRAY_LENGTH3 - 1;
var nativeFloor4 = Math.floor;
var nativeMin11 = Math.min;
function baseSortedIndexBy(array, value, iteratee2, retHighest) {
  var low = 0, high = array == null ? 0 : array.length;
  if (high === 0) {
    return 0;
  }
  value = iteratee2(value);
  var valIsNaN = value !== value, valIsNull = value === null, valIsSymbol = isSymbol_default(value), valIsUndefined = value === void 0;
  while (low < high) {
    var mid = nativeFloor4((low + high) / 2), computed = iteratee2(array[mid]), othIsDefined = computed !== void 0, othIsNull = computed === null, othIsReflexive = computed === computed, othIsSymbol = isSymbol_default(computed);
    if (valIsNaN) {
      var setLow = retHighest || othIsReflexive;
    } else if (valIsUndefined) {
      setLow = othIsReflexive && (retHighest || othIsDefined);
    } else if (valIsNull) {
      setLow = othIsReflexive && othIsDefined && (retHighest || !othIsNull);
    } else if (valIsSymbol) {
      setLow = othIsReflexive && othIsDefined && !othIsNull && (retHighest || !othIsSymbol);
    } else if (othIsNull || othIsSymbol) {
      setLow = false;
    } else {
      setLow = retHighest ? computed <= value : computed < value;
    }
    if (setLow) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return nativeMin11(high, MAX_ARRAY_INDEX);
}
var baseSortedIndexBy_default = baseSortedIndexBy;

// node_modules/lodash-es/_baseSortedIndex.js
var MAX_ARRAY_LENGTH4 = 4294967295;
var HALF_MAX_ARRAY_LENGTH = MAX_ARRAY_LENGTH4 >>> 1;
function baseSortedIndex(array, value, retHighest) {
  var low = 0, high = array == null ? low : array.length;
  if (typeof value == "number" && value === value && high <= HALF_MAX_ARRAY_LENGTH) {
    while (low < high) {
      var mid = low + high >>> 1, computed = array[mid];
      if (computed !== null && !isSymbol_default(computed) && (retHighest ? computed <= value : computed < value)) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return high;
  }
  return baseSortedIndexBy_default(array, value, identity_default, retHighest);
}
var baseSortedIndex_default = baseSortedIndex;

// node_modules/lodash-es/sortedIndex.js
function sortedIndex(array, value) {
  return baseSortedIndex_default(array, value);
}
var sortedIndex_default = sortedIndex;

// node_modules/lodash-es/sortedIndexBy.js
function sortedIndexBy(array, value, iteratee2) {
  return baseSortedIndexBy_default(array, value, baseIteratee_default(iteratee2, 2));
}
var sortedIndexBy_default = sortedIndexBy;

// node_modules/lodash-es/sortedIndexOf.js
function sortedIndexOf(array, value) {
  var length = array == null ? 0 : array.length;
  if (length) {
    var index = baseSortedIndex_default(array, value);
    if (index < length && eq_default(array[index], value)) {
      return index;
    }
  }
  return -1;
}
var sortedIndexOf_default = sortedIndexOf;

// node_modules/lodash-es/sortedLastIndex.js
function sortedLastIndex(array, value) {
  return baseSortedIndex_default(array, value, true);
}
var sortedLastIndex_default = sortedLastIndex;

// node_modules/lodash-es/sortedLastIndexBy.js
function sortedLastIndexBy(array, value, iteratee2) {
  return baseSortedIndexBy_default(array, value, baseIteratee_default(iteratee2, 2), true);
}
var sortedLastIndexBy_default = sortedLastIndexBy;

// node_modules/lodash-es/sortedLastIndexOf.js
function sortedLastIndexOf(array, value) {
  var length = array == null ? 0 : array.length;
  if (length) {
    var index = baseSortedIndex_default(array, value, true) - 1;
    if (eq_default(array[index], value)) {
      return index;
    }
  }
  return -1;
}
var sortedLastIndexOf_default = sortedLastIndexOf;

// node_modules/lodash-es/_baseSortedUniq.js
function baseSortedUniq(array, iteratee2) {
  var index = -1, length = array.length, resIndex = 0, result2 = [];
  while (++index < length) {
    var value = array[index], computed = iteratee2 ? iteratee2(value) : value;
    if (!index || !eq_default(computed, seen)) {
      var seen = computed;
      result2[resIndex++] = value === 0 ? 0 : value;
    }
  }
  return result2;
}
var baseSortedUniq_default = baseSortedUniq;

// node_modules/lodash-es/sortedUniq.js
function sortedUniq(array) {
  return array && array.length ? baseSortedUniq_default(array) : [];
}
var sortedUniq_default = sortedUniq;

// node_modules/lodash-es/sortedUniqBy.js
function sortedUniqBy(array, iteratee2) {
  return array && array.length ? baseSortedUniq_default(array, baseIteratee_default(iteratee2, 2)) : [];
}
var sortedUniqBy_default = sortedUniqBy;

// node_modules/lodash-es/split.js
var MAX_ARRAY_LENGTH5 = 4294967295;
function split(string, separator, limit) {
  if (limit && typeof limit != "number" && isIterateeCall_default(string, separator, limit)) {
    separator = limit = void 0;
  }
  limit = limit === void 0 ? MAX_ARRAY_LENGTH5 : limit >>> 0;
  if (!limit) {
    return [];
  }
  string = toString_default(string);
  if (string && (typeof separator == "string" || separator != null && !isRegExp_default(separator))) {
    separator = baseToString_default(separator);
    if (!separator && hasUnicode_default(string)) {
      return castSlice_default(stringToArray_default(string), 0, limit);
    }
  }
  return string.split(separator, limit);
}
var split_default = split;

// node_modules/lodash-es/spread.js
var FUNC_ERROR_TEXT11 = "Expected a function";
var nativeMax14 = Math.max;
function spread(func, start) {
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT11);
  }
  start = start == null ? 0 : nativeMax14(toInteger_default(start), 0);
  return baseRest_default(function(args) {
    var array = args[start], otherArgs = castSlice_default(args, 0, start);
    if (array) {
      arrayPush_default(otherArgs, array);
    }
    return apply_default(func, this, otherArgs);
  });
}
var spread_default = spread;

// node_modules/lodash-es/startCase.js
var startCase = createCompounder_default(function(result2, word, index) {
  return result2 + (index ? " " : "") + upperFirst_default(word);
});
var startCase_default = startCase;

// node_modules/lodash-es/startsWith.js
function startsWith(string, target, position) {
  string = toString_default(string);
  position = position == null ? 0 : baseClamp_default(toInteger_default(position), 0, string.length);
  target = baseToString_default(target);
  return string.slice(position, position + target.length) == target;
}
var startsWith_default = startsWith;

// node_modules/lodash-es/stubObject.js
function stubObject() {
  return {};
}
var stubObject_default = stubObject;

// node_modules/lodash-es/stubString.js
function stubString() {
  return "";
}
var stubString_default = stubString;

// node_modules/lodash-es/stubTrue.js
function stubTrue() {
  return true;
}
var stubTrue_default = stubTrue;

// node_modules/lodash-es/subtract.js
var subtract = createMathOperation_default(function(minuend, subtrahend) {
  return minuend - subtrahend;
}, 0);
var subtract_default = subtract;

// node_modules/lodash-es/sum.js
function sum(array) {
  return array && array.length ? baseSum_default(array, identity_default) : 0;
}
var sum_default = sum;

// node_modules/lodash-es/sumBy.js
function sumBy(array, iteratee2) {
  return array && array.length ? baseSum_default(array, baseIteratee_default(iteratee2, 2)) : 0;
}
var sumBy_default = sumBy;

// node_modules/lodash-es/tail.js
function tail(array) {
  var length = array == null ? 0 : array.length;
  return length ? baseSlice_default(array, 1, length) : [];
}
var tail_default = tail;

// node_modules/lodash-es/take.js
function take(array, n, guard) {
  if (!(array && array.length)) {
    return [];
  }
  n = guard || n === void 0 ? 1 : toInteger_default(n);
  return baseSlice_default(array, 0, n < 0 ? 0 : n);
}
var take_default = take;

// node_modules/lodash-es/takeRight.js
function takeRight(array, n, guard) {
  var length = array == null ? 0 : array.length;
  if (!length) {
    return [];
  }
  n = guard || n === void 0 ? 1 : toInteger_default(n);
  n = length - n;
  return baseSlice_default(array, n < 0 ? 0 : n, length);
}
var takeRight_default = takeRight;

// node_modules/lodash-es/takeRightWhile.js
function takeRightWhile(array, predicate) {
  return array && array.length ? baseWhile_default(array, baseIteratee_default(predicate, 3), false, true) : [];
}
var takeRightWhile_default = takeRightWhile;

// node_modules/lodash-es/takeWhile.js
function takeWhile(array, predicate) {
  return array && array.length ? baseWhile_default(array, baseIteratee_default(predicate, 3)) : [];
}
var takeWhile_default = takeWhile;

// node_modules/lodash-es/tap.js
function tap(value, interceptor) {
  interceptor(value);
  return value;
}
var tap_default = tap;

// node_modules/lodash-es/_customDefaultsAssignIn.js
var objectProto27 = Object.prototype;
var hasOwnProperty23 = objectProto27.hasOwnProperty;
function customDefaultsAssignIn(objValue, srcValue, key, object) {
  if (objValue === void 0 || eq_default(objValue, objectProto27[key]) && !hasOwnProperty23.call(object, key)) {
    return srcValue;
  }
  return objValue;
}
var customDefaultsAssignIn_default = customDefaultsAssignIn;

// node_modules/lodash-es/_escapeStringChar.js
var stringEscapes = {
  "\\": "\\",
  "'": "'",
  "\n": "n",
  "\r": "r",
  "\u2028": "u2028",
  "\u2029": "u2029"
};
function escapeStringChar(chr) {
  return "\\" + stringEscapes[chr];
}
var escapeStringChar_default = escapeStringChar;

// node_modules/lodash-es/_reInterpolate.js
var reInterpolate = /<%=([\s\S]+?)%>/g;
var reInterpolate_default = reInterpolate;

// node_modules/lodash-es/_reEscape.js
var reEscape = /<%-([\s\S]+?)%>/g;
var reEscape_default = reEscape;

// node_modules/lodash-es/_reEvaluate.js
var reEvaluate = /<%([\s\S]+?)%>/g;
var reEvaluate_default = reEvaluate;

// node_modules/lodash-es/templateSettings.js
var templateSettings = {
  /**
   * Used to detect `data` property values to be HTML-escaped.
   *
   * @memberOf _.templateSettings
   * @type {RegExp}
   */
  "escape": reEscape_default,
  /**
   * Used to detect code to be evaluated.
   *
   * @memberOf _.templateSettings
   * @type {RegExp}
   */
  "evaluate": reEvaluate_default,
  /**
   * Used to detect `data` property values to inject.
   *
   * @memberOf _.templateSettings
   * @type {RegExp}
   */
  "interpolate": reInterpolate_default,
  /**
   * Used to reference the data object in the template text.
   *
   * @memberOf _.templateSettings
   * @type {string}
   */
  "variable": "",
  /**
   * Used to import variables into the compiled template.
   *
   * @memberOf _.templateSettings
   * @type {Object}
   */
  "imports": {
    /**
     * A reference to the `lodash` function.
     *
     * @memberOf _.templateSettings.imports
     * @type {Function}
     */
    "_": { "escape": escape_default }
  }
};
var templateSettings_default = templateSettings;

// node_modules/lodash-es/template.js
var INVALID_TEMPL_VAR_ERROR_TEXT = "Invalid `variable` option passed into `_.template`";
var reEmptyStringLeading = /\b__p \+= '';/g;
var reEmptyStringMiddle = /\b(__p \+=) '' \+/g;
var reEmptyStringTrailing = /(__e\(.*?\)|\b__t\)) \+\n'';/g;
var reForbiddenIdentifierChars = /[()=,{}\[\]\/\s]/;
var reEsTemplate = /\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g;
var reNoMatch = /($^)/;
var reUnescapedString = /['\n\r\u2028\u2029\\]/g;
var objectProto28 = Object.prototype;
var hasOwnProperty24 = objectProto28.hasOwnProperty;
function template(string, options, guard) {
  var settings = templateSettings_default.imports._.templateSettings || templateSettings_default;
  if (guard && isIterateeCall_default(string, options, guard)) {
    options = void 0;
  }
  string = toString_default(string);
  options = assignInWith_default({}, options, settings, customDefaultsAssignIn_default);
  var imports = assignInWith_default({}, options.imports, settings.imports, customDefaultsAssignIn_default), importsKeys = keys_default(imports), importsValues = baseValues_default(imports, importsKeys);
  var isEscaping, isEvaluating, index = 0, interpolate = options.interpolate || reNoMatch, source = "__p += '";
  var reDelimiters = RegExp(
    (options.escape || reNoMatch).source + "|" + interpolate.source + "|" + (interpolate === reInterpolate_default ? reEsTemplate : reNoMatch).source + "|" + (options.evaluate || reNoMatch).source + "|$",
    "g"
  );
  var sourceURL = hasOwnProperty24.call(options, "sourceURL") ? "//# sourceURL=" + (options.sourceURL + "").replace(/\s/g, " ") + "\n" : "";
  string.replace(reDelimiters, function(match, escapeValue, interpolateValue, esTemplateValue, evaluateValue, offset) {
    interpolateValue || (interpolateValue = esTemplateValue);
    source += string.slice(index, offset).replace(reUnescapedString, escapeStringChar_default);
    if (escapeValue) {
      isEscaping = true;
      source += "' +\n__e(" + escapeValue + ") +\n'";
    }
    if (evaluateValue) {
      isEvaluating = true;
      source += "';\n" + evaluateValue + ";\n__p += '";
    }
    if (interpolateValue) {
      source += "' +\n((__t = (" + interpolateValue + ")) == null ? '' : __t) +\n'";
    }
    index = offset + match.length;
    return match;
  });
  source += "';\n";
  var variable = hasOwnProperty24.call(options, "variable") && options.variable;
  if (!variable) {
    source = "with (obj) {\n" + source + "\n}\n";
  } else if (reForbiddenIdentifierChars.test(variable)) {
    throw new Error(INVALID_TEMPL_VAR_ERROR_TEXT);
  }
  source = (isEvaluating ? source.replace(reEmptyStringLeading, "") : source).replace(reEmptyStringMiddle, "$1").replace(reEmptyStringTrailing, "$1;");
  source = "function(" + (variable || "obj") + ") {\n" + (variable ? "" : "obj || (obj = {});\n") + "var __t, __p = ''" + (isEscaping ? ", __e = _.escape" : "") + (isEvaluating ? ", __j = Array.prototype.join;\nfunction print() { __p += __j.call(arguments, '') }\n" : ";\n") + source + "return __p\n}";
  var result2 = attempt_default(function() {
    return Function(importsKeys, sourceURL + "return " + source).apply(void 0, importsValues);
  });
  result2.source = source;
  if (isError_default(result2)) {
    throw result2;
  }
  return result2;
}
var template_default = template;

// node_modules/lodash-es/throttle.js
var FUNC_ERROR_TEXT12 = "Expected a function";
function throttle(func, wait, options) {
  var leading = true, trailing = true;
  if (typeof func != "function") {
    throw new TypeError(FUNC_ERROR_TEXT12);
  }
  if (isObject_default(options)) {
    leading = "leading" in options ? !!options.leading : leading;
    trailing = "trailing" in options ? !!options.trailing : trailing;
  }
  return debounce_default(func, wait, {
    "leading": leading,
    "maxWait": wait,
    "trailing": trailing
  });
}
var throttle_default = throttle;

// node_modules/lodash-es/thru.js
function thru(value, interceptor) {
  return interceptor(value);
}
var thru_default = thru;

// node_modules/lodash-es/times.js
var MAX_SAFE_INTEGER5 = 9007199254740991;
var MAX_ARRAY_LENGTH6 = 4294967295;
var nativeMin12 = Math.min;
function times(n, iteratee2) {
  n = toInteger_default(n);
  if (n < 1 || n > MAX_SAFE_INTEGER5) {
    return [];
  }
  var index = MAX_ARRAY_LENGTH6, length = nativeMin12(n, MAX_ARRAY_LENGTH6);
  iteratee2 = castFunction_default(iteratee2);
  n -= MAX_ARRAY_LENGTH6;
  var result2 = baseTimes_default(length, iteratee2);
  while (++index < n) {
    iteratee2(index);
  }
  return result2;
}
var times_default = times;

// node_modules/lodash-es/toIterator.js
function wrapperToIterator() {
  return this;
}
var toIterator_default = wrapperToIterator;

// node_modules/lodash-es/_baseWrapperValue.js
function baseWrapperValue(value, actions) {
  var result2 = value;
  if (result2 instanceof LazyWrapper_default) {
    result2 = result2.value();
  }
  return arrayReduce_default(actions, function(result3, action) {
    return action.func.apply(action.thisArg, arrayPush_default([result3], action.args));
  }, result2);
}
var baseWrapperValue_default = baseWrapperValue;

// node_modules/lodash-es/wrapperValue.js
function wrapperValue() {
  return baseWrapperValue_default(this.__wrapped__, this.__actions__);
}
var wrapperValue_default = wrapperValue;

// node_modules/lodash-es/toLower.js
function toLower(value) {
  return toString_default(value).toLowerCase();
}
var toLower_default = toLower;

// node_modules/lodash-es/toPath.js
function toPath(value) {
  if (isArray_default(value)) {
    return arrayMap_default(value, toKey_default);
  }
  return isSymbol_default(value) ? [value] : copyArray_default(stringToPath_default(toString_default(value)));
}
var toPath_default = toPath;

// node_modules/lodash-es/toSafeInteger.js
var MAX_SAFE_INTEGER6 = 9007199254740991;
function toSafeInteger(value) {
  return value ? baseClamp_default(toInteger_default(value), -MAX_SAFE_INTEGER6, MAX_SAFE_INTEGER6) : value === 0 ? value : 0;
}
var toSafeInteger_default = toSafeInteger;

// node_modules/lodash-es/toUpper.js
function toUpper(value) {
  return toString_default(value).toUpperCase();
}
var toUpper_default = toUpper;

// node_modules/lodash-es/transform.js
function transform(object, iteratee2, accumulator) {
  var isArr = isArray_default(object), isArrLike = isArr || isBuffer_default(object) || isTypedArray_default(object);
  iteratee2 = baseIteratee_default(iteratee2, 4);
  if (accumulator == null) {
    var Ctor = object && object.constructor;
    if (isArrLike) {
      accumulator = isArr ? new Ctor() : [];
    } else if (isObject_default(object)) {
      accumulator = isFunction_default(Ctor) ? baseCreate_default(getPrototype_default(object)) : {};
    } else {
      accumulator = {};
    }
  }
  (isArrLike ? arrayEach_default : baseForOwn_default)(object, function(value, index, object2) {
    return iteratee2(accumulator, value, index, object2);
  });
  return accumulator;
}
var transform_default = transform;

// node_modules/lodash-es/_charsEndIndex.js
function charsEndIndex(strSymbols, chrSymbols) {
  var index = strSymbols.length;
  while (index-- && baseIndexOf_default(chrSymbols, strSymbols[index], 0) > -1) {
  }
  return index;
}
var charsEndIndex_default = charsEndIndex;

// node_modules/lodash-es/_charsStartIndex.js
function charsStartIndex(strSymbols, chrSymbols) {
  var index = -1, length = strSymbols.length;
  while (++index < length && baseIndexOf_default(chrSymbols, strSymbols[index], 0) > -1) {
  }
  return index;
}
var charsStartIndex_default = charsStartIndex;

// node_modules/lodash-es/trim.js
function trim(string, chars, guard) {
  string = toString_default(string);
  if (string && (guard || chars === void 0)) {
    return baseTrim_default(string);
  }
  if (!string || !(chars = baseToString_default(chars))) {
    return string;
  }
  var strSymbols = stringToArray_default(string), chrSymbols = stringToArray_default(chars), start = charsStartIndex_default(strSymbols, chrSymbols), end = charsEndIndex_default(strSymbols, chrSymbols) + 1;
  return castSlice_default(strSymbols, start, end).join("");
}
var trim_default = trim;

// node_modules/lodash-es/trimEnd.js
function trimEnd(string, chars, guard) {
  string = toString_default(string);
  if (string && (guard || chars === void 0)) {
    return string.slice(0, trimmedEndIndex_default(string) + 1);
  }
  if (!string || !(chars = baseToString_default(chars))) {
    return string;
  }
  var strSymbols = stringToArray_default(string), end = charsEndIndex_default(strSymbols, stringToArray_default(chars)) + 1;
  return castSlice_default(strSymbols, 0, end).join("");
}
var trimEnd_default = trimEnd;

// node_modules/lodash-es/trimStart.js
var reTrimStart3 = /^\s+/;
function trimStart(string, chars, guard) {
  string = toString_default(string);
  if (string && (guard || chars === void 0)) {
    return string.replace(reTrimStart3, "");
  }
  if (!string || !(chars = baseToString_default(chars))) {
    return string;
  }
  var strSymbols = stringToArray_default(string), start = charsStartIndex_default(strSymbols, stringToArray_default(chars));
  return castSlice_default(strSymbols, start).join("");
}
var trimStart_default = trimStart;

// node_modules/lodash-es/truncate.js
var DEFAULT_TRUNC_LENGTH = 30;
var DEFAULT_TRUNC_OMISSION = "...";
var reFlags2 = /\w*$/;
function truncate(string, options) {
  var length = DEFAULT_TRUNC_LENGTH, omission = DEFAULT_TRUNC_OMISSION;
  if (isObject_default(options)) {
    var separator = "separator" in options ? options.separator : separator;
    length = "length" in options ? toInteger_default(options.length) : length;
    omission = "omission" in options ? baseToString_default(options.omission) : omission;
  }
  string = toString_default(string);
  var strLength = string.length;
  if (hasUnicode_default(string)) {
    var strSymbols = stringToArray_default(string);
    strLength = strSymbols.length;
  }
  if (length >= strLength) {
    return string;
  }
  var end = length - stringSize_default(omission);
  if (end < 1) {
    return omission;
  }
  var result2 = strSymbols ? castSlice_default(strSymbols, 0, end).join("") : string.slice(0, end);
  if (separator === void 0) {
    return result2 + omission;
  }
  if (strSymbols) {
    end += result2.length - end;
  }
  if (isRegExp_default(separator)) {
    if (string.slice(end).search(separator)) {
      var match, substring = result2;
      if (!separator.global) {
        separator = RegExp(separator.source, toString_default(reFlags2.exec(separator)) + "g");
      }
      separator.lastIndex = 0;
      while (match = separator.exec(substring)) {
        var newEnd = match.index;
      }
      result2 = result2.slice(0, newEnd === void 0 ? end : newEnd);
    }
  } else if (string.indexOf(baseToString_default(separator), end) != end) {
    var index = result2.lastIndexOf(separator);
    if (index > -1) {
      result2 = result2.slice(0, index);
    }
  }
  return result2 + omission;
}
var truncate_default = truncate;

// node_modules/lodash-es/unary.js
function unary(func) {
  return ary_default(func, 1);
}
var unary_default = unary;

// node_modules/lodash-es/_unescapeHtmlChar.js
var htmlUnescapes = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'"
};
var unescapeHtmlChar = basePropertyOf_default(htmlUnescapes);
var unescapeHtmlChar_default = unescapeHtmlChar;

// node_modules/lodash-es/unescape.js
var reEscapedHtml = /&(?:amp|lt|gt|quot|#39);/g;
var reHasEscapedHtml = RegExp(reEscapedHtml.source);
function unescape(string) {
  string = toString_default(string);
  return string && reHasEscapedHtml.test(string) ? string.replace(reEscapedHtml, unescapeHtmlChar_default) : string;
}
var unescape_default = unescape;

// node_modules/lodash-es/_createSet.js
var INFINITY6 = 1 / 0;
var createSet = !(Set_default && 1 / setToArray_default(new Set_default([, -0]))[1] == INFINITY6) ? noop_default : function(values2) {
  return new Set_default(values2);
};
var createSet_default = createSet;

// node_modules/lodash-es/_baseUniq.js
var LARGE_ARRAY_SIZE3 = 200;
function baseUniq(array, iteratee2, comparator) {
  var index = -1, includes2 = arrayIncludes_default, length = array.length, isCommon = true, result2 = [], seen = result2;
  if (comparator) {
    isCommon = false;
    includes2 = arrayIncludesWith_default;
  } else if (length >= LARGE_ARRAY_SIZE3) {
    var set2 = iteratee2 ? null : createSet_default(array);
    if (set2) {
      return setToArray_default(set2);
    }
    isCommon = false;
    includes2 = cacheHas_default;
    seen = new SetCache_default();
  } else {
    seen = iteratee2 ? [] : result2;
  }
  outer:
    while (++index < length) {
      var value = array[index], computed = iteratee2 ? iteratee2(value) : value;
      value = comparator || value !== 0 ? value : 0;
      if (isCommon && computed === computed) {
        var seenIndex = seen.length;
        while (seenIndex--) {
          if (seen[seenIndex] === computed) {
            continue outer;
          }
        }
        if (iteratee2) {
          seen.push(computed);
        }
        result2.push(value);
      } else if (!includes2(seen, computed, comparator)) {
        if (seen !== result2) {
          seen.push(computed);
        }
        result2.push(value);
      }
    }
  return result2;
}
var baseUniq_default = baseUniq;

// node_modules/lodash-es/union.js
var union = baseRest_default(function(arrays) {
  return baseUniq_default(baseFlatten_default(arrays, 1, isArrayLikeObject_default, true));
});
var union_default = union;

// node_modules/lodash-es/unionBy.js
var unionBy = baseRest_default(function(arrays) {
  var iteratee2 = last_default(arrays);
  if (isArrayLikeObject_default(iteratee2)) {
    iteratee2 = void 0;
  }
  return baseUniq_default(baseFlatten_default(arrays, 1, isArrayLikeObject_default, true), baseIteratee_default(iteratee2, 2));
});
var unionBy_default = unionBy;

// node_modules/lodash-es/unionWith.js
var unionWith = baseRest_default(function(arrays) {
  var comparator = last_default(arrays);
  comparator = typeof comparator == "function" ? comparator : void 0;
  return baseUniq_default(baseFlatten_default(arrays, 1, isArrayLikeObject_default, true), void 0, comparator);
});
var unionWith_default = unionWith;

// node_modules/lodash-es/uniq.js
function uniq(array) {
  return array && array.length ? baseUniq_default(array) : [];
}
var uniq_default = uniq;

// node_modules/lodash-es/uniqBy.js
function uniqBy(array, iteratee2) {
  return array && array.length ? baseUniq_default(array, baseIteratee_default(iteratee2, 2)) : [];
}
var uniqBy_default = uniqBy;

// node_modules/lodash-es/uniqWith.js
function uniqWith(array, comparator) {
  comparator = typeof comparator == "function" ? comparator : void 0;
  return array && array.length ? baseUniq_default(array, void 0, comparator) : [];
}
var uniqWith_default = uniqWith;

// node_modules/lodash-es/uniqueId.js
var idCounter = 0;
function uniqueId(prefix) {
  var id = ++idCounter;
  return toString_default(prefix) + id;
}
var uniqueId_default = uniqueId;

// node_modules/lodash-es/unset.js
function unset(object, path) {
  return object == null ? true : baseUnset_default(object, path);
}
var unset_default = unset;

// node_modules/lodash-es/unzip.js
var nativeMax15 = Math.max;
function unzip(array) {
  if (!(array && array.length)) {
    return [];
  }
  var length = 0;
  array = arrayFilter_default(array, function(group) {
    if (isArrayLikeObject_default(group)) {
      length = nativeMax15(group.length, length);
      return true;
    }
  });
  return baseTimes_default(length, function(index) {
    return arrayMap_default(array, baseProperty_default(index));
  });
}
var unzip_default = unzip;

// node_modules/lodash-es/unzipWith.js
function unzipWith(array, iteratee2) {
  if (!(array && array.length)) {
    return [];
  }
  var result2 = unzip_default(array);
  if (iteratee2 == null) {
    return result2;
  }
  return arrayMap_default(result2, function(group) {
    return apply_default(iteratee2, void 0, group);
  });
}
var unzipWith_default = unzipWith;

// node_modules/lodash-es/_baseUpdate.js
function baseUpdate(object, path, updater, customizer) {
  return baseSet_default(object, path, updater(baseGet_default(object, path)), customizer);
}
var baseUpdate_default = baseUpdate;

// node_modules/lodash-es/update.js
function update(object, path, updater) {
  return object == null ? object : baseUpdate_default(object, path, castFunction_default(updater));
}
var update_default = update;

// node_modules/lodash-es/updateWith.js
function updateWith(object, path, updater, customizer) {
  customizer = typeof customizer == "function" ? customizer : void 0;
  return object == null ? object : baseUpdate_default(object, path, castFunction_default(updater), customizer);
}
var updateWith_default = updateWith;

// node_modules/lodash-es/upperCase.js
var upperCase = createCompounder_default(function(result2, word, index) {
  return result2 + (index ? " " : "") + word.toUpperCase();
});
var upperCase_default = upperCase;

// node_modules/lodash-es/valuesIn.js
function valuesIn(object) {
  return object == null ? [] : baseValues_default(object, keysIn_default(object));
}
var valuesIn_default = valuesIn;

// node_modules/lodash-es/without.js
var without = baseRest_default(function(array, values2) {
  return isArrayLikeObject_default(array) ? baseDifference_default(array, values2) : [];
});
var without_default = without;

// node_modules/lodash-es/wrap.js
function wrap(value, wrapper) {
  return partial_default(castFunction_default(wrapper), value);
}
var wrap_default = wrap;

// node_modules/lodash-es/wrapperAt.js
var wrapperAt = flatRest_default(function(paths) {
  var length = paths.length, start = length ? paths[0] : 0, value = this.__wrapped__, interceptor = function(object) {
    return baseAt_default(object, paths);
  };
  if (length > 1 || this.__actions__.length || !(value instanceof LazyWrapper_default) || !isIndex_default(start)) {
    return this.thru(interceptor);
  }
  value = value.slice(start, +start + (length ? 1 : 0));
  value.__actions__.push({
    "func": thru_default,
    "args": [interceptor],
    "thisArg": void 0
  });
  return new LodashWrapper_default(value, this.__chain__).thru(function(array) {
    if (length && !array.length) {
      array.push(void 0);
    }
    return array;
  });
});
var wrapperAt_default = wrapperAt;

// node_modules/lodash-es/wrapperChain.js
function wrapperChain() {
  return chain_default(this);
}
var wrapperChain_default = wrapperChain;

// node_modules/lodash-es/wrapperReverse.js
function wrapperReverse() {
  var value = this.__wrapped__;
  if (value instanceof LazyWrapper_default) {
    var wrapped = value;
    if (this.__actions__.length) {
      wrapped = new LazyWrapper_default(this);
    }
    wrapped = wrapped.reverse();
    wrapped.__actions__.push({
      "func": thru_default,
      "args": [reverse_default],
      "thisArg": void 0
    });
    return new LodashWrapper_default(wrapped, this.__chain__);
  }
  return this.thru(reverse_default);
}
var wrapperReverse_default = wrapperReverse;

// node_modules/lodash-es/_baseXor.js
function baseXor(arrays, iteratee2, comparator) {
  var length = arrays.length;
  if (length < 2) {
    return length ? baseUniq_default(arrays[0]) : [];
  }
  var index = -1, result2 = Array(length);
  while (++index < length) {
    var array = arrays[index], othIndex = -1;
    while (++othIndex < length) {
      if (othIndex != index) {
        result2[index] = baseDifference_default(result2[index] || array, arrays[othIndex], iteratee2, comparator);
      }
    }
  }
  return baseUniq_default(baseFlatten_default(result2, 1), iteratee2, comparator);
}
var baseXor_default = baseXor;

// node_modules/lodash-es/xor.js
var xor = baseRest_default(function(arrays) {
  return baseXor_default(arrayFilter_default(arrays, isArrayLikeObject_default));
});
var xor_default = xor;

// node_modules/lodash-es/xorBy.js
var xorBy = baseRest_default(function(arrays) {
  var iteratee2 = last_default(arrays);
  if (isArrayLikeObject_default(iteratee2)) {
    iteratee2 = void 0;
  }
  return baseXor_default(arrayFilter_default(arrays, isArrayLikeObject_default), baseIteratee_default(iteratee2, 2));
});
var xorBy_default = xorBy;

// node_modules/lodash-es/xorWith.js
var xorWith = baseRest_default(function(arrays) {
  var comparator = last_default(arrays);
  comparator = typeof comparator == "function" ? comparator : void 0;
  return baseXor_default(arrayFilter_default(arrays, isArrayLikeObject_default), void 0, comparator);
});
var xorWith_default = xorWith;

// node_modules/lodash-es/zip.js
var zip = baseRest_default(unzip_default);
var zip_default = zip;

// node_modules/lodash-es/_baseZipObject.js
function baseZipObject(props, values2, assignFunc) {
  var index = -1, length = props.length, valsLength = values2.length, result2 = {};
  while (++index < length) {
    var value = index < valsLength ? values2[index] : void 0;
    assignFunc(result2, props[index], value);
  }
  return result2;
}
var baseZipObject_default = baseZipObject;

// node_modules/lodash-es/zipObject.js
function zipObject(props, values2) {
  return baseZipObject_default(props || [], values2 || [], assignValue_default);
}
var zipObject_default = zipObject;

// node_modules/lodash-es/zipObjectDeep.js
function zipObjectDeep(props, values2) {
  return baseZipObject_default(props || [], values2 || [], baseSet_default);
}
var zipObjectDeep_default = zipObjectDeep;

// node_modules/lodash-es/zipWith.js
var zipWith = baseRest_default(function(arrays) {
  var length = arrays.length, iteratee2 = length > 1 ? arrays[length - 1] : void 0;
  iteratee2 = typeof iteratee2 == "function" ? (arrays.pop(), iteratee2) : void 0;
  return unzipWith_default(arrays, iteratee2);
});
var zipWith_default = zipWith;

// node_modules/lodash-es/array.default.js
var array_default_default = {
  chunk: chunk_default,
  compact: compact_default,
  concat: concat_default,
  difference: difference_default,
  differenceBy: differenceBy_default,
  differenceWith: differenceWith_default,
  drop: drop_default,
  dropRight: dropRight_default,
  dropRightWhile: dropRightWhile_default,
  dropWhile: dropWhile_default,
  fill: fill_default,
  findIndex: findIndex_default,
  findLastIndex: findLastIndex_default,
  first: head_default,
  flatten: flatten_default,
  flattenDeep: flattenDeep_default,
  flattenDepth: flattenDepth_default,
  fromPairs: fromPairs_default,
  head: head_default,
  indexOf: indexOf_default,
  initial: initial_default,
  intersection: intersection_default,
  intersectionBy: intersectionBy_default,
  intersectionWith: intersectionWith_default,
  join: join_default,
  last: last_default,
  lastIndexOf: lastIndexOf_default,
  nth: nth_default,
  pull: pull_default,
  pullAll: pullAll_default,
  pullAllBy: pullAllBy_default,
  pullAllWith: pullAllWith_default,
  pullAt: pullAt_default,
  remove: remove_default,
  reverse: reverse_default,
  slice: slice_default,
  sortedIndex: sortedIndex_default,
  sortedIndexBy: sortedIndexBy_default,
  sortedIndexOf: sortedIndexOf_default,
  sortedLastIndex: sortedLastIndex_default,
  sortedLastIndexBy: sortedLastIndexBy_default,
  sortedLastIndexOf: sortedLastIndexOf_default,
  sortedUniq: sortedUniq_default,
  sortedUniqBy: sortedUniqBy_default,
  tail: tail_default,
  take: take_default,
  takeRight: takeRight_default,
  takeRightWhile: takeRightWhile_default,
  takeWhile: takeWhile_default,
  union: union_default,
  unionBy: unionBy_default,
  unionWith: unionWith_default,
  uniq: uniq_default,
  uniqBy: uniqBy_default,
  uniqWith: uniqWith_default,
  unzip: unzip_default,
  unzipWith: unzipWith_default,
  without: without_default,
  xor: xor_default,
  xorBy: xorBy_default,
  xorWith: xorWith_default,
  zip: zip_default,
  zipObject: zipObject_default,
  zipObjectDeep: zipObjectDeep_default,
  zipWith: zipWith_default
};

// node_modules/lodash-es/collection.default.js
var collection_default_default = {
  countBy: countBy_default,
  each: forEach_default,
  eachRight: forEachRight_default,
  every: every_default,
  filter: filter_default,
  find: find_default,
  findLast: findLast_default,
  flatMap: flatMap_default,
  flatMapDeep: flatMapDeep_default,
  flatMapDepth: flatMapDepth_default,
  forEach: forEach_default,
  forEachRight: forEachRight_default,
  groupBy: groupBy_default,
  includes: includes_default,
  invokeMap: invokeMap_default,
  keyBy: keyBy_default,
  map: map_default,
  orderBy: orderBy_default,
  partition: partition_default,
  reduce: reduce_default,
  reduceRight: reduceRight_default,
  reject: reject_default,
  sample: sample_default,
  sampleSize: sampleSize_default,
  shuffle: shuffle_default,
  size: size_default,
  some: some_default,
  sortBy: sortBy_default
};

// node_modules/lodash-es/date.default.js
var date_default_default = {
  now: now_default
};

// node_modules/lodash-es/function.default.js
var function_default_default = {
  after: after_default,
  ary: ary_default,
  before: before_default,
  bind: bind_default,
  bindKey: bindKey_default,
  curry: curry_default,
  curryRight: curryRight_default,
  debounce: debounce_default,
  defer: defer_default,
  delay: delay_default,
  flip: flip_default,
  memoize: memoize_default,
  negate: negate_default,
  once: once_default,
  overArgs: overArgs_default,
  partial: partial_default,
  partialRight: partialRight_default,
  rearg: rearg_default,
  rest: rest_default,
  spread: spread_default,
  throttle: throttle_default,
  unary: unary_default,
  wrap: wrap_default
};

// node_modules/lodash-es/lang.default.js
var lang_default_default = {
  castArray: castArray_default,
  clone: clone_default,
  cloneDeep: cloneDeep_default,
  cloneDeepWith: cloneDeepWith_default,
  cloneWith: cloneWith_default,
  conformsTo: conformsTo_default,
  eq: eq_default,
  gt: gt_default,
  gte: gte_default,
  isArguments: isArguments_default,
  isArray: isArray_default,
  isArrayBuffer: isArrayBuffer_default,
  isArrayLike: isArrayLike_default,
  isArrayLikeObject: isArrayLikeObject_default,
  isBoolean: isBoolean_default,
  isBuffer: isBuffer_default,
  isDate: isDate_default,
  isElement: isElement_default,
  isEmpty: isEmpty_default,
  isEqual: isEqual_default,
  isEqualWith: isEqualWith_default,
  isError: isError_default,
  isFinite: isFinite_default,
  isFunction: isFunction_default,
  isInteger: isInteger_default,
  isLength: isLength_default,
  isMap: isMap_default,
  isMatch: isMatch_default,
  isMatchWith: isMatchWith_default,
  isNaN: isNaN_default,
  isNative: isNative_default,
  isNil: isNil_default,
  isNull: isNull_default,
  isNumber: isNumber_default,
  isObject: isObject_default,
  isObjectLike: isObjectLike_default,
  isPlainObject: isPlainObject_default,
  isRegExp: isRegExp_default,
  isSafeInteger: isSafeInteger_default,
  isSet: isSet_default,
  isString: isString_default,
  isSymbol: isSymbol_default,
  isTypedArray: isTypedArray_default,
  isUndefined: isUndefined_default,
  isWeakMap: isWeakMap_default,
  isWeakSet: isWeakSet_default,
  lt: lt_default,
  lte: lte_default,
  toArray: toArray_default,
  toFinite: toFinite_default,
  toInteger: toInteger_default,
  toLength: toLength_default,
  toNumber: toNumber_default,
  toPlainObject: toPlainObject_default,
  toSafeInteger: toSafeInteger_default,
  toString: toString_default
};

// node_modules/lodash-es/math.default.js
var math_default_default = {
  add: add_default,
  ceil: ceil_default,
  divide: divide_default,
  floor: floor_default,
  max: max_default,
  maxBy: maxBy_default,
  mean: mean_default,
  meanBy: meanBy_default,
  min: min_default,
  minBy: minBy_default,
  multiply: multiply_default,
  round: round_default,
  subtract: subtract_default,
  sum: sum_default,
  sumBy: sumBy_default
};

// node_modules/lodash-es/number.default.js
var number_default_default = {
  clamp: clamp_default,
  inRange: inRange_default,
  random: random_default
};

// node_modules/lodash-es/object.default.js
var object_default_default = {
  assign: assign_default,
  assignIn: assignIn_default,
  assignInWith: assignInWith_default,
  assignWith: assignWith_default,
  at: at_default,
  create: create_default,
  defaults: defaults_default,
  defaultsDeep: defaultsDeep_default,
  entries: toPairs_default,
  entriesIn: toPairsIn_default,
  extend: assignIn_default,
  extendWith: assignInWith_default,
  findKey: findKey_default,
  findLastKey: findLastKey_default,
  forIn: forIn_default,
  forInRight: forInRight_default,
  forOwn: forOwn_default,
  forOwnRight: forOwnRight_default,
  functions: functions_default,
  functionsIn: functionsIn_default,
  get: get_default,
  has: has_default,
  hasIn: hasIn_default,
  invert: invert_default,
  invertBy: invertBy_default,
  invoke: invoke_default,
  keys: keys_default,
  keysIn: keysIn_default,
  mapKeys: mapKeys_default,
  mapValues: mapValues_default,
  merge: merge_default,
  mergeWith: mergeWith_default,
  omit: omit_default,
  omitBy: omitBy_default,
  pick: pick_default,
  pickBy: pickBy_default,
  result: result_default,
  set: set_default,
  setWith: setWith_default,
  toPairs: toPairs_default,
  toPairsIn: toPairsIn_default,
  transform: transform_default,
  unset: unset_default,
  update: update_default,
  updateWith: updateWith_default,
  values: values_default,
  valuesIn: valuesIn_default
};

// node_modules/lodash-es/seq.default.js
var seq_default_default = {
  at: wrapperAt_default,
  chain: chain_default,
  commit: commit_default,
  lodash: wrapperLodash_default,
  next: next_default,
  plant: plant_default,
  reverse: wrapperReverse_default,
  tap: tap_default,
  thru: thru_default,
  toIterator: toIterator_default,
  toJSON: wrapperValue_default,
  value: wrapperValue_default,
  valueOf: wrapperValue_default,
  wrapperChain: wrapperChain_default
};

// node_modules/lodash-es/string.default.js
var string_default_default = {
  camelCase: camelCase_default,
  capitalize: capitalize_default,
  deburr: deburr_default,
  endsWith: endsWith_default,
  escape: escape_default,
  escapeRegExp: escapeRegExp_default,
  kebabCase: kebabCase_default,
  lowerCase: lowerCase_default,
  lowerFirst: lowerFirst_default,
  pad: pad_default,
  padEnd: padEnd_default,
  padStart: padStart_default,
  parseInt: parseInt_default,
  repeat: repeat_default,
  replace: replace_default,
  snakeCase: snakeCase_default,
  split: split_default,
  startCase: startCase_default,
  startsWith: startsWith_default,
  template: template_default,
  templateSettings: templateSettings_default,
  toLower: toLower_default,
  toUpper: toUpper_default,
  trim: trim_default,
  trimEnd: trimEnd_default,
  trimStart: trimStart_default,
  truncate: truncate_default,
  unescape: unescape_default,
  upperCase: upperCase_default,
  upperFirst: upperFirst_default,
  words: words_default
};

// node_modules/lodash-es/util.default.js
var util_default_default = {
  attempt: attempt_default,
  bindAll: bindAll_default,
  cond: cond_default,
  conforms: conforms_default,
  constant: constant_default,
  defaultTo: defaultTo_default,
  flow: flow_default,
  flowRight: flowRight_default,
  identity: identity_default,
  iteratee: iteratee_default,
  matches: matches_default,
  matchesProperty: matchesProperty_default,
  method: method_default,
  methodOf: methodOf_default,
  mixin: mixin_default,
  noop: noop_default,
  nthArg: nthArg_default,
  over: over_default,
  overEvery: overEvery_default,
  overSome: overSome_default,
  property: property_default,
  propertyOf: propertyOf_default,
  range: range_default,
  rangeRight: rangeRight_default,
  stubArray: stubArray_default,
  stubFalse: stubFalse_default,
  stubObject: stubObject_default,
  stubString: stubString_default,
  stubTrue: stubTrue_default,
  times: times_default,
  toPath: toPath_default,
  uniqueId: uniqueId_default
};

// node_modules/lodash-es/_lazyClone.js
function lazyClone() {
  var result2 = new LazyWrapper_default(this.__wrapped__);
  result2.__actions__ = copyArray_default(this.__actions__);
  result2.__dir__ = this.__dir__;
  result2.__filtered__ = this.__filtered__;
  result2.__iteratees__ = copyArray_default(this.__iteratees__);
  result2.__takeCount__ = this.__takeCount__;
  result2.__views__ = copyArray_default(this.__views__);
  return result2;
}
var lazyClone_default = lazyClone;

// node_modules/lodash-es/_lazyReverse.js
function lazyReverse() {
  if (this.__filtered__) {
    var result2 = new LazyWrapper_default(this);
    result2.__dir__ = -1;
    result2.__filtered__ = true;
  } else {
    result2 = this.clone();
    result2.__dir__ *= -1;
  }
  return result2;
}
var lazyReverse_default = lazyReverse;

// node_modules/lodash-es/_getView.js
var nativeMax16 = Math.max;
var nativeMin13 = Math.min;
function getView(start, end, transforms) {
  var index = -1, length = transforms.length;
  while (++index < length) {
    var data = transforms[index], size2 = data.size;
    switch (data.type) {
      case "drop":
        start += size2;
        break;
      case "dropRight":
        end -= size2;
        break;
      case "take":
        end = nativeMin13(end, start + size2);
        break;
      case "takeRight":
        start = nativeMax16(start, end - size2);
        break;
    }
  }
  return { "start": start, "end": end };
}
var getView_default = getView;

// node_modules/lodash-es/_lazyValue.js
var LAZY_FILTER_FLAG = 1;
var LAZY_MAP_FLAG = 2;
var nativeMin14 = Math.min;
function lazyValue() {
  var array = this.__wrapped__.value(), dir = this.__dir__, isArr = isArray_default(array), isRight = dir < 0, arrLength = isArr ? array.length : 0, view = getView_default(0, arrLength, this.__views__), start = view.start, end = view.end, length = end - start, index = isRight ? end : start - 1, iteratees = this.__iteratees__, iterLength = iteratees.length, resIndex = 0, takeCount = nativeMin14(length, this.__takeCount__);
  if (!isArr || !isRight && arrLength == length && takeCount == length) {
    return baseWrapperValue_default(array, this.__actions__);
  }
  var result2 = [];
  outer:
    while (length-- && resIndex < takeCount) {
      index += dir;
      var iterIndex = -1, value = array[index];
      while (++iterIndex < iterLength) {
        var data = iteratees[iterIndex], iteratee2 = data.iteratee, type = data.type, computed = iteratee2(value);
        if (type == LAZY_MAP_FLAG) {
          value = computed;
        } else if (!computed) {
          if (type == LAZY_FILTER_FLAG) {
            continue outer;
          } else {
            break outer;
          }
        }
      }
      result2[resIndex++] = value;
    }
  return result2;
}
var lazyValue_default = lazyValue;

// node_modules/lodash-es/lodash.default.js
var VERSION = "4.17.21";
var WRAP_BIND_KEY_FLAG7 = 2;
var LAZY_FILTER_FLAG2 = 1;
var LAZY_WHILE_FLAG = 3;
var MAX_ARRAY_LENGTH7 = 4294967295;
var arrayProto6 = Array.prototype;
var objectProto29 = Object.prototype;
var hasOwnProperty25 = objectProto29.hasOwnProperty;
var symIterator2 = Symbol_default ? Symbol_default.iterator : void 0;
var nativeMax17 = Math.max;
var nativeMin15 = Math.min;
var mixin2 = /* @__PURE__ */ function(func) {
  return function(object, source, options) {
    if (options == null) {
      var isObj = isObject_default(source), props = isObj && keys_default(source), methodNames = props && props.length && baseFunctions_default(source, props);
      if (!(methodNames ? methodNames.length : isObj)) {
        options = source;
        source = object;
        object = this;
      }
    }
    return func(object, source, options);
  };
}(mixin_default);
wrapperLodash_default.after = function_default_default.after;
wrapperLodash_default.ary = function_default_default.ary;
wrapperLodash_default.assign = object_default_default.assign;
wrapperLodash_default.assignIn = object_default_default.assignIn;
wrapperLodash_default.assignInWith = object_default_default.assignInWith;
wrapperLodash_default.assignWith = object_default_default.assignWith;
wrapperLodash_default.at = object_default_default.at;
wrapperLodash_default.before = function_default_default.before;
wrapperLodash_default.bind = function_default_default.bind;
wrapperLodash_default.bindAll = util_default_default.bindAll;
wrapperLodash_default.bindKey = function_default_default.bindKey;
wrapperLodash_default.castArray = lang_default_default.castArray;
wrapperLodash_default.chain = seq_default_default.chain;
wrapperLodash_default.chunk = array_default_default.chunk;
wrapperLodash_default.compact = array_default_default.compact;
wrapperLodash_default.concat = array_default_default.concat;
wrapperLodash_default.cond = util_default_default.cond;
wrapperLodash_default.conforms = util_default_default.conforms;
wrapperLodash_default.constant = util_default_default.constant;
wrapperLodash_default.countBy = collection_default_default.countBy;
wrapperLodash_default.create = object_default_default.create;
wrapperLodash_default.curry = function_default_default.curry;
wrapperLodash_default.curryRight = function_default_default.curryRight;
wrapperLodash_default.debounce = function_default_default.debounce;
wrapperLodash_default.defaults = object_default_default.defaults;
wrapperLodash_default.defaultsDeep = object_default_default.defaultsDeep;
wrapperLodash_default.defer = function_default_default.defer;
wrapperLodash_default.delay = function_default_default.delay;
wrapperLodash_default.difference = array_default_default.difference;
wrapperLodash_default.differenceBy = array_default_default.differenceBy;
wrapperLodash_default.differenceWith = array_default_default.differenceWith;
wrapperLodash_default.drop = array_default_default.drop;
wrapperLodash_default.dropRight = array_default_default.dropRight;
wrapperLodash_default.dropRightWhile = array_default_default.dropRightWhile;
wrapperLodash_default.dropWhile = array_default_default.dropWhile;
wrapperLodash_default.fill = array_default_default.fill;
wrapperLodash_default.filter = collection_default_default.filter;
wrapperLodash_default.flatMap = collection_default_default.flatMap;
wrapperLodash_default.flatMapDeep = collection_default_default.flatMapDeep;
wrapperLodash_default.flatMapDepth = collection_default_default.flatMapDepth;
wrapperLodash_default.flatten = array_default_default.flatten;
wrapperLodash_default.flattenDeep = array_default_default.flattenDeep;
wrapperLodash_default.flattenDepth = array_default_default.flattenDepth;
wrapperLodash_default.flip = function_default_default.flip;
wrapperLodash_default.flow = util_default_default.flow;
wrapperLodash_default.flowRight = util_default_default.flowRight;
wrapperLodash_default.fromPairs = array_default_default.fromPairs;
wrapperLodash_default.functions = object_default_default.functions;
wrapperLodash_default.functionsIn = object_default_default.functionsIn;
wrapperLodash_default.groupBy = collection_default_default.groupBy;
wrapperLodash_default.initial = array_default_default.initial;
wrapperLodash_default.intersection = array_default_default.intersection;
wrapperLodash_default.intersectionBy = array_default_default.intersectionBy;
wrapperLodash_default.intersectionWith = array_default_default.intersectionWith;
wrapperLodash_default.invert = object_default_default.invert;
wrapperLodash_default.invertBy = object_default_default.invertBy;
wrapperLodash_default.invokeMap = collection_default_default.invokeMap;
wrapperLodash_default.iteratee = util_default_default.iteratee;
wrapperLodash_default.keyBy = collection_default_default.keyBy;
wrapperLodash_default.keys = keys_default;
wrapperLodash_default.keysIn = object_default_default.keysIn;
wrapperLodash_default.map = collection_default_default.map;
wrapperLodash_default.mapKeys = object_default_default.mapKeys;
wrapperLodash_default.mapValues = object_default_default.mapValues;
wrapperLodash_default.matches = util_default_default.matches;
wrapperLodash_default.matchesProperty = util_default_default.matchesProperty;
wrapperLodash_default.memoize = function_default_default.memoize;
wrapperLodash_default.merge = object_default_default.merge;
wrapperLodash_default.mergeWith = object_default_default.mergeWith;
wrapperLodash_default.method = util_default_default.method;
wrapperLodash_default.methodOf = util_default_default.methodOf;
wrapperLodash_default.mixin = mixin2;
wrapperLodash_default.negate = negate_default;
wrapperLodash_default.nthArg = util_default_default.nthArg;
wrapperLodash_default.omit = object_default_default.omit;
wrapperLodash_default.omitBy = object_default_default.omitBy;
wrapperLodash_default.once = function_default_default.once;
wrapperLodash_default.orderBy = collection_default_default.orderBy;
wrapperLodash_default.over = util_default_default.over;
wrapperLodash_default.overArgs = function_default_default.overArgs;
wrapperLodash_default.overEvery = util_default_default.overEvery;
wrapperLodash_default.overSome = util_default_default.overSome;
wrapperLodash_default.partial = function_default_default.partial;
wrapperLodash_default.partialRight = function_default_default.partialRight;
wrapperLodash_default.partition = collection_default_default.partition;
wrapperLodash_default.pick = object_default_default.pick;
wrapperLodash_default.pickBy = object_default_default.pickBy;
wrapperLodash_default.property = util_default_default.property;
wrapperLodash_default.propertyOf = util_default_default.propertyOf;
wrapperLodash_default.pull = array_default_default.pull;
wrapperLodash_default.pullAll = array_default_default.pullAll;
wrapperLodash_default.pullAllBy = array_default_default.pullAllBy;
wrapperLodash_default.pullAllWith = array_default_default.pullAllWith;
wrapperLodash_default.pullAt = array_default_default.pullAt;
wrapperLodash_default.range = util_default_default.range;
wrapperLodash_default.rangeRight = util_default_default.rangeRight;
wrapperLodash_default.rearg = function_default_default.rearg;
wrapperLodash_default.reject = collection_default_default.reject;
wrapperLodash_default.remove = array_default_default.remove;
wrapperLodash_default.rest = function_default_default.rest;
wrapperLodash_default.reverse = array_default_default.reverse;
wrapperLodash_default.sampleSize = collection_default_default.sampleSize;
wrapperLodash_default.set = object_default_default.set;
wrapperLodash_default.setWith = object_default_default.setWith;
wrapperLodash_default.shuffle = collection_default_default.shuffle;
wrapperLodash_default.slice = array_default_default.slice;
wrapperLodash_default.sortBy = collection_default_default.sortBy;
wrapperLodash_default.sortedUniq = array_default_default.sortedUniq;
wrapperLodash_default.sortedUniqBy = array_default_default.sortedUniqBy;
wrapperLodash_default.split = string_default_default.split;
wrapperLodash_default.spread = function_default_default.spread;
wrapperLodash_default.tail = array_default_default.tail;
wrapperLodash_default.take = array_default_default.take;
wrapperLodash_default.takeRight = array_default_default.takeRight;
wrapperLodash_default.takeRightWhile = array_default_default.takeRightWhile;
wrapperLodash_default.takeWhile = array_default_default.takeWhile;
wrapperLodash_default.tap = seq_default_default.tap;
wrapperLodash_default.throttle = function_default_default.throttle;
wrapperLodash_default.thru = thru_default;
wrapperLodash_default.toArray = lang_default_default.toArray;
wrapperLodash_default.toPairs = object_default_default.toPairs;
wrapperLodash_default.toPairsIn = object_default_default.toPairsIn;
wrapperLodash_default.toPath = util_default_default.toPath;
wrapperLodash_default.toPlainObject = lang_default_default.toPlainObject;
wrapperLodash_default.transform = object_default_default.transform;
wrapperLodash_default.unary = function_default_default.unary;
wrapperLodash_default.union = array_default_default.union;
wrapperLodash_default.unionBy = array_default_default.unionBy;
wrapperLodash_default.unionWith = array_default_default.unionWith;
wrapperLodash_default.uniq = array_default_default.uniq;
wrapperLodash_default.uniqBy = array_default_default.uniqBy;
wrapperLodash_default.uniqWith = array_default_default.uniqWith;
wrapperLodash_default.unset = object_default_default.unset;
wrapperLodash_default.unzip = array_default_default.unzip;
wrapperLodash_default.unzipWith = array_default_default.unzipWith;
wrapperLodash_default.update = object_default_default.update;
wrapperLodash_default.updateWith = object_default_default.updateWith;
wrapperLodash_default.values = object_default_default.values;
wrapperLodash_default.valuesIn = object_default_default.valuesIn;
wrapperLodash_default.without = array_default_default.without;
wrapperLodash_default.words = string_default_default.words;
wrapperLodash_default.wrap = function_default_default.wrap;
wrapperLodash_default.xor = array_default_default.xor;
wrapperLodash_default.xorBy = array_default_default.xorBy;
wrapperLodash_default.xorWith = array_default_default.xorWith;
wrapperLodash_default.zip = array_default_default.zip;
wrapperLodash_default.zipObject = array_default_default.zipObject;
wrapperLodash_default.zipObjectDeep = array_default_default.zipObjectDeep;
wrapperLodash_default.zipWith = array_default_default.zipWith;
wrapperLodash_default.entries = object_default_default.toPairs;
wrapperLodash_default.entriesIn = object_default_default.toPairsIn;
wrapperLodash_default.extend = object_default_default.assignIn;
wrapperLodash_default.extendWith = object_default_default.assignInWith;
mixin2(wrapperLodash_default, wrapperLodash_default);
wrapperLodash_default.add = math_default_default.add;
wrapperLodash_default.attempt = util_default_default.attempt;
wrapperLodash_default.camelCase = string_default_default.camelCase;
wrapperLodash_default.capitalize = string_default_default.capitalize;
wrapperLodash_default.ceil = math_default_default.ceil;
wrapperLodash_default.clamp = number_default_default.clamp;
wrapperLodash_default.clone = lang_default_default.clone;
wrapperLodash_default.cloneDeep = lang_default_default.cloneDeep;
wrapperLodash_default.cloneDeepWith = lang_default_default.cloneDeepWith;
wrapperLodash_default.cloneWith = lang_default_default.cloneWith;
wrapperLodash_default.conformsTo = lang_default_default.conformsTo;
wrapperLodash_default.deburr = string_default_default.deburr;
wrapperLodash_default.defaultTo = util_default_default.defaultTo;
wrapperLodash_default.divide = math_default_default.divide;
wrapperLodash_default.endsWith = string_default_default.endsWith;
wrapperLodash_default.eq = lang_default_default.eq;
wrapperLodash_default.escape = string_default_default.escape;
wrapperLodash_default.escapeRegExp = string_default_default.escapeRegExp;
wrapperLodash_default.every = collection_default_default.every;
wrapperLodash_default.find = collection_default_default.find;
wrapperLodash_default.findIndex = array_default_default.findIndex;
wrapperLodash_default.findKey = object_default_default.findKey;
wrapperLodash_default.findLast = collection_default_default.findLast;
wrapperLodash_default.findLastIndex = array_default_default.findLastIndex;
wrapperLodash_default.findLastKey = object_default_default.findLastKey;
wrapperLodash_default.floor = math_default_default.floor;
wrapperLodash_default.forEach = collection_default_default.forEach;
wrapperLodash_default.forEachRight = collection_default_default.forEachRight;
wrapperLodash_default.forIn = object_default_default.forIn;
wrapperLodash_default.forInRight = object_default_default.forInRight;
wrapperLodash_default.forOwn = object_default_default.forOwn;
wrapperLodash_default.forOwnRight = object_default_default.forOwnRight;
wrapperLodash_default.get = object_default_default.get;
wrapperLodash_default.gt = lang_default_default.gt;
wrapperLodash_default.gte = lang_default_default.gte;
wrapperLodash_default.has = object_default_default.has;
wrapperLodash_default.hasIn = object_default_default.hasIn;
wrapperLodash_default.head = array_default_default.head;
wrapperLodash_default.identity = identity_default;
wrapperLodash_default.includes = collection_default_default.includes;
wrapperLodash_default.indexOf = array_default_default.indexOf;
wrapperLodash_default.inRange = number_default_default.inRange;
wrapperLodash_default.invoke = object_default_default.invoke;
wrapperLodash_default.isArguments = lang_default_default.isArguments;
wrapperLodash_default.isArray = isArray_default;
wrapperLodash_default.isArrayBuffer = lang_default_default.isArrayBuffer;
wrapperLodash_default.isArrayLike = lang_default_default.isArrayLike;
wrapperLodash_default.isArrayLikeObject = lang_default_default.isArrayLikeObject;
wrapperLodash_default.isBoolean = lang_default_default.isBoolean;
wrapperLodash_default.isBuffer = lang_default_default.isBuffer;
wrapperLodash_default.isDate = lang_default_default.isDate;
wrapperLodash_default.isElement = lang_default_default.isElement;
wrapperLodash_default.isEmpty = lang_default_default.isEmpty;
wrapperLodash_default.isEqual = lang_default_default.isEqual;
wrapperLodash_default.isEqualWith = lang_default_default.isEqualWith;
wrapperLodash_default.isError = lang_default_default.isError;
wrapperLodash_default.isFinite = lang_default_default.isFinite;
wrapperLodash_default.isFunction = lang_default_default.isFunction;
wrapperLodash_default.isInteger = lang_default_default.isInteger;
wrapperLodash_default.isLength = lang_default_default.isLength;
wrapperLodash_default.isMap = lang_default_default.isMap;
wrapperLodash_default.isMatch = lang_default_default.isMatch;
wrapperLodash_default.isMatchWith = lang_default_default.isMatchWith;
wrapperLodash_default.isNaN = lang_default_default.isNaN;
wrapperLodash_default.isNative = lang_default_default.isNative;
wrapperLodash_default.isNil = lang_default_default.isNil;
wrapperLodash_default.isNull = lang_default_default.isNull;
wrapperLodash_default.isNumber = lang_default_default.isNumber;
wrapperLodash_default.isObject = isObject_default;
wrapperLodash_default.isObjectLike = lang_default_default.isObjectLike;
wrapperLodash_default.isPlainObject = lang_default_default.isPlainObject;
wrapperLodash_default.isRegExp = lang_default_default.isRegExp;
wrapperLodash_default.isSafeInteger = lang_default_default.isSafeInteger;
wrapperLodash_default.isSet = lang_default_default.isSet;
wrapperLodash_default.isString = lang_default_default.isString;
wrapperLodash_default.isSymbol = lang_default_default.isSymbol;
wrapperLodash_default.isTypedArray = lang_default_default.isTypedArray;
wrapperLodash_default.isUndefined = lang_default_default.isUndefined;
wrapperLodash_default.isWeakMap = lang_default_default.isWeakMap;
wrapperLodash_default.isWeakSet = lang_default_default.isWeakSet;
wrapperLodash_default.join = array_default_default.join;
wrapperLodash_default.kebabCase = string_default_default.kebabCase;
wrapperLodash_default.last = last_default;
wrapperLodash_default.lastIndexOf = array_default_default.lastIndexOf;
wrapperLodash_default.lowerCase = string_default_default.lowerCase;
wrapperLodash_default.lowerFirst = string_default_default.lowerFirst;
wrapperLodash_default.lt = lang_default_default.lt;
wrapperLodash_default.lte = lang_default_default.lte;
wrapperLodash_default.max = math_default_default.max;
wrapperLodash_default.maxBy = math_default_default.maxBy;
wrapperLodash_default.mean = math_default_default.mean;
wrapperLodash_default.meanBy = math_default_default.meanBy;
wrapperLodash_default.min = math_default_default.min;
wrapperLodash_default.minBy = math_default_default.minBy;
wrapperLodash_default.stubArray = util_default_default.stubArray;
wrapperLodash_default.stubFalse = util_default_default.stubFalse;
wrapperLodash_default.stubObject = util_default_default.stubObject;
wrapperLodash_default.stubString = util_default_default.stubString;
wrapperLodash_default.stubTrue = util_default_default.stubTrue;
wrapperLodash_default.multiply = math_default_default.multiply;
wrapperLodash_default.nth = array_default_default.nth;
wrapperLodash_default.noop = util_default_default.noop;
wrapperLodash_default.now = date_default_default.now;
wrapperLodash_default.pad = string_default_default.pad;
wrapperLodash_default.padEnd = string_default_default.padEnd;
wrapperLodash_default.padStart = string_default_default.padStart;
wrapperLodash_default.parseInt = string_default_default.parseInt;
wrapperLodash_default.random = number_default_default.random;
wrapperLodash_default.reduce = collection_default_default.reduce;
wrapperLodash_default.reduceRight = collection_default_default.reduceRight;
wrapperLodash_default.repeat = string_default_default.repeat;
wrapperLodash_default.replace = string_default_default.replace;
wrapperLodash_default.result = object_default_default.result;
wrapperLodash_default.round = math_default_default.round;
wrapperLodash_default.sample = collection_default_default.sample;
wrapperLodash_default.size = collection_default_default.size;
wrapperLodash_default.snakeCase = string_default_default.snakeCase;
wrapperLodash_default.some = collection_default_default.some;
wrapperLodash_default.sortedIndex = array_default_default.sortedIndex;
wrapperLodash_default.sortedIndexBy = array_default_default.sortedIndexBy;
wrapperLodash_default.sortedIndexOf = array_default_default.sortedIndexOf;
wrapperLodash_default.sortedLastIndex = array_default_default.sortedLastIndex;
wrapperLodash_default.sortedLastIndexBy = array_default_default.sortedLastIndexBy;
wrapperLodash_default.sortedLastIndexOf = array_default_default.sortedLastIndexOf;
wrapperLodash_default.startCase = string_default_default.startCase;
wrapperLodash_default.startsWith = string_default_default.startsWith;
wrapperLodash_default.subtract = math_default_default.subtract;
wrapperLodash_default.sum = math_default_default.sum;
wrapperLodash_default.sumBy = math_default_default.sumBy;
wrapperLodash_default.template = string_default_default.template;
wrapperLodash_default.times = util_default_default.times;
wrapperLodash_default.toFinite = lang_default_default.toFinite;
wrapperLodash_default.toInteger = toInteger_default;
wrapperLodash_default.toLength = lang_default_default.toLength;
wrapperLodash_default.toLower = string_default_default.toLower;
wrapperLodash_default.toNumber = lang_default_default.toNumber;
wrapperLodash_default.toSafeInteger = lang_default_default.toSafeInteger;
wrapperLodash_default.toString = lang_default_default.toString;
wrapperLodash_default.toUpper = string_default_default.toUpper;
wrapperLodash_default.trim = string_default_default.trim;
wrapperLodash_default.trimEnd = string_default_default.trimEnd;
wrapperLodash_default.trimStart = string_default_default.trimStart;
wrapperLodash_default.truncate = string_default_default.truncate;
wrapperLodash_default.unescape = string_default_default.unescape;
wrapperLodash_default.uniqueId = util_default_default.uniqueId;
wrapperLodash_default.upperCase = string_default_default.upperCase;
wrapperLodash_default.upperFirst = string_default_default.upperFirst;
wrapperLodash_default.each = collection_default_default.forEach;
wrapperLodash_default.eachRight = collection_default_default.forEachRight;
wrapperLodash_default.first = array_default_default.head;
mixin2(wrapperLodash_default, function() {
  var source = {};
  baseForOwn_default(wrapperLodash_default, function(func, methodName) {
    if (!hasOwnProperty25.call(wrapperLodash_default.prototype, methodName)) {
      source[methodName] = func;
    }
  });
  return source;
}(), { "chain": false });
wrapperLodash_default.VERSION = VERSION;
(wrapperLodash_default.templateSettings = string_default_default.templateSettings).imports._ = wrapperLodash_default;
arrayEach_default(["bind", "bindKey", "curry", "curryRight", "partial", "partialRight"], function(methodName) {
  wrapperLodash_default[methodName].placeholder = wrapperLodash_default;
});
arrayEach_default(["drop", "take"], function(methodName, index) {
  LazyWrapper_default.prototype[methodName] = function(n) {
    n = n === void 0 ? 1 : nativeMax17(toInteger_default(n), 0);
    var result2 = this.__filtered__ && !index ? new LazyWrapper_default(this) : this.clone();
    if (result2.__filtered__) {
      result2.__takeCount__ = nativeMin15(n, result2.__takeCount__);
    } else {
      result2.__views__.push({
        "size": nativeMin15(n, MAX_ARRAY_LENGTH7),
        "type": methodName + (result2.__dir__ < 0 ? "Right" : "")
      });
    }
    return result2;
  };
  LazyWrapper_default.prototype[methodName + "Right"] = function(n) {
    return this.reverse()[methodName](n).reverse();
  };
});
arrayEach_default(["filter", "map", "takeWhile"], function(methodName, index) {
  var type = index + 1, isFilter = type == LAZY_FILTER_FLAG2 || type == LAZY_WHILE_FLAG;
  LazyWrapper_default.prototype[methodName] = function(iteratee2) {
    var result2 = this.clone();
    result2.__iteratees__.push({
      "iteratee": baseIteratee_default(iteratee2, 3),
      "type": type
    });
    result2.__filtered__ = result2.__filtered__ || isFilter;
    return result2;
  };
});
arrayEach_default(["head", "last"], function(methodName, index) {
  var takeName = "take" + (index ? "Right" : "");
  LazyWrapper_default.prototype[methodName] = function() {
    return this[takeName](1).value()[0];
  };
});
arrayEach_default(["initial", "tail"], function(methodName, index) {
  var dropName = "drop" + (index ? "" : "Right");
  LazyWrapper_default.prototype[methodName] = function() {
    return this.__filtered__ ? new LazyWrapper_default(this) : this[dropName](1);
  };
});
LazyWrapper_default.prototype.compact = function() {
  return this.filter(identity_default);
};
LazyWrapper_default.prototype.find = function(predicate) {
  return this.filter(predicate).head();
};
LazyWrapper_default.prototype.findLast = function(predicate) {
  return this.reverse().find(predicate);
};
LazyWrapper_default.prototype.invokeMap = baseRest_default(function(path, args) {
  if (typeof path == "function") {
    return new LazyWrapper_default(this);
  }
  return this.map(function(value) {
    return baseInvoke_default(value, path, args);
  });
});
LazyWrapper_default.prototype.reject = function(predicate) {
  return this.filter(negate_default(baseIteratee_default(predicate)));
};
LazyWrapper_default.prototype.slice = function(start, end) {
  start = toInteger_default(start);
  var result2 = this;
  if (result2.__filtered__ && (start > 0 || end < 0)) {
    return new LazyWrapper_default(result2);
  }
  if (start < 0) {
    result2 = result2.takeRight(-start);
  } else if (start) {
    result2 = result2.drop(start);
  }
  if (end !== void 0) {
    end = toInteger_default(end);
    result2 = end < 0 ? result2.dropRight(-end) : result2.take(end - start);
  }
  return result2;
};
LazyWrapper_default.prototype.takeRightWhile = function(predicate) {
  return this.reverse().takeWhile(predicate).reverse();
};
LazyWrapper_default.prototype.toArray = function() {
  return this.take(MAX_ARRAY_LENGTH7);
};
baseForOwn_default(LazyWrapper_default.prototype, function(func, methodName) {
  var checkIteratee = /^(?:filter|find|map|reject)|While$/.test(methodName), isTaker = /^(?:head|last)$/.test(methodName), lodashFunc = wrapperLodash_default[isTaker ? "take" + (methodName == "last" ? "Right" : "") : methodName], retUnwrapped = isTaker || /^find/.test(methodName);
  if (!lodashFunc) {
    return;
  }
  wrapperLodash_default.prototype[methodName] = function() {
    var value = this.__wrapped__, args = isTaker ? [1] : arguments, isLazy = value instanceof LazyWrapper_default, iteratee2 = args[0], useLazy = isLazy || isArray_default(value);
    var interceptor = function(value2) {
      var result3 = lodashFunc.apply(wrapperLodash_default, arrayPush_default([value2], args));
      return isTaker && chainAll ? result3[0] : result3;
    };
    if (useLazy && checkIteratee && typeof iteratee2 == "function" && iteratee2.length != 1) {
      isLazy = useLazy = false;
    }
    var chainAll = this.__chain__, isHybrid = !!this.__actions__.length, isUnwrapped = retUnwrapped && !chainAll, onlyLazy = isLazy && !isHybrid;
    if (!retUnwrapped && useLazy) {
      value = onlyLazy ? value : new LazyWrapper_default(this);
      var result2 = func.apply(value, args);
      result2.__actions__.push({ "func": thru_default, "args": [interceptor], "thisArg": void 0 });
      return new LodashWrapper_default(result2, chainAll);
    }
    if (isUnwrapped && onlyLazy) {
      return func.apply(this, args);
    }
    result2 = this.thru(interceptor);
    return isUnwrapped ? isTaker ? result2.value()[0] : result2.value() : result2;
  };
});
arrayEach_default(["pop", "push", "shift", "sort", "splice", "unshift"], function(methodName) {
  var func = arrayProto6[methodName], chainName = /^(?:push|sort|unshift)$/.test(methodName) ? "tap" : "thru", retUnwrapped = /^(?:pop|shift)$/.test(methodName);
  wrapperLodash_default.prototype[methodName] = function() {
    var args = arguments;
    if (retUnwrapped && !this.__chain__) {
      var value = this.value();
      return func.apply(isArray_default(value) ? value : [], args);
    }
    return this[chainName](function(value2) {
      return func.apply(isArray_default(value2) ? value2 : [], args);
    });
  };
});
baseForOwn_default(LazyWrapper_default.prototype, function(func, methodName) {
  var lodashFunc = wrapperLodash_default[methodName];
  if (lodashFunc) {
    var key = lodashFunc.name + "";
    if (!hasOwnProperty25.call(realNames_default, key)) {
      realNames_default[key] = [];
    }
    realNames_default[key].push({ "name": methodName, "func": lodashFunc });
  }
});
realNames_default[createHybrid_default(void 0, WRAP_BIND_KEY_FLAG7).name] = [{
  "name": "wrapper",
  "func": void 0
}];
LazyWrapper_default.prototype.clone = lazyClone_default;
LazyWrapper_default.prototype.reverse = lazyReverse_default;
LazyWrapper_default.prototype.value = lazyValue_default;
wrapperLodash_default.prototype.at = seq_default_default.at;
wrapperLodash_default.prototype.chain = seq_default_default.wrapperChain;
wrapperLodash_default.prototype.commit = seq_default_default.commit;
wrapperLodash_default.prototype.next = seq_default_default.next;
wrapperLodash_default.prototype.plant = seq_default_default.plant;
wrapperLodash_default.prototype.reverse = seq_default_default.reverse;
wrapperLodash_default.prototype.toJSON = wrapperLodash_default.prototype.valueOf = wrapperLodash_default.prototype.value = seq_default_default.value;
wrapperLodash_default.prototype.first = wrapperLodash_default.prototype.head;
if (symIterator2) {
  wrapperLodash_default.prototype[symIterator2] = seq_default_default.toIterator;
}

// node_modules/@chevrotain/utils/lib/src/print.js
function PRINT_ERROR(msg) {
  if (console && console.error) {
    console.error(`Error: ${msg}`);
  }
}
function PRINT_WARNING(msg) {
  if (console && console.warn) {
    console.warn(`Warning: ${msg}`);
  }
}

// node_modules/@chevrotain/utils/lib/src/timer.js
function timer(func) {
  const start = (/* @__PURE__ */ new Date()).getTime();
  const val = func();
  const end = (/* @__PURE__ */ new Date()).getTime();
  const total = end - start;
  return { time: total, value: val };
}

// node_modules/@chevrotain/utils/lib/src/to-fast-properties.js
function toFastProperties(toBecomeFast) {
  function FakeConstructor() {
  }
  FakeConstructor.prototype = toBecomeFast;
  const fakeInstance = new FakeConstructor();
  function fakeAccess() {
    return typeof fakeInstance.bar;
  }
  fakeAccess();
  fakeAccess();
  if (1)
    return toBecomeFast;
  (0, eval)(toBecomeFast);
}

// node_modules/@chevrotain/gast/lib/src/model.js
function tokenLabel(tokType) {
  if (hasTokenLabel(tokType)) {
    return tokType.LABEL;
  } else {
    return tokType.name;
  }
}
function hasTokenLabel(obj) {
  return isString_default(obj.LABEL) && obj.LABEL !== "";
}
var AbstractProduction = class {
  get definition() {
    return this._definition;
  }
  set definition(value) {
    this._definition = value;
  }
  constructor(_definition) {
    this._definition = _definition;
  }
  accept(visitor2) {
    visitor2.visit(this);
    forEach_default(this.definition, (prod) => {
      prod.accept(visitor2);
    });
  }
};
var NonTerminal = class extends AbstractProduction {
  constructor(options) {
    super([]);
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
  set definition(definition) {
  }
  get definition() {
    if (this.referencedRule !== void 0) {
      return this.referencedRule.definition;
    }
    return [];
  }
  accept(visitor2) {
    visitor2.visit(this);
  }
};
var Rule = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.orgText = "";
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var Alternative = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.ignoreAmbiguities = false;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var Option = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var RepetitionMandatory = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var RepetitionMandatoryWithSeparator = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var Repetition = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var RepetitionWithSeparator = class extends AbstractProduction {
  constructor(options) {
    super(options.definition);
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var Alternation = class extends AbstractProduction {
  get definition() {
    return this._definition;
  }
  set definition(value) {
    this._definition = value;
  }
  constructor(options) {
    super(options.definition);
    this.idx = 1;
    this.ignoreAmbiguities = false;
    this.hasPredicates = false;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
};
var Terminal = class {
  constructor(options) {
    this.idx = 1;
    assign_default(this, pickBy_default(options, (v) => v !== void 0));
  }
  accept(visitor2) {
    visitor2.visit(this);
  }
};
function serializeGrammar(topRules) {
  return map_default(topRules, serializeProduction);
}
function serializeProduction(node) {
  function convertDefinition(definition) {
    return map_default(definition, serializeProduction);
  }
  if (node instanceof NonTerminal) {
    const serializedNonTerminal = {
      type: "NonTerminal",
      name: node.nonTerminalName,
      idx: node.idx
    };
    if (isString_default(node.label)) {
      serializedNonTerminal.label = node.label;
    }
    return serializedNonTerminal;
  } else if (node instanceof Alternative) {
    return {
      type: "Alternative",
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof Option) {
    return {
      type: "Option",
      idx: node.idx,
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof RepetitionMandatory) {
    return {
      type: "RepetitionMandatory",
      idx: node.idx,
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof RepetitionMandatoryWithSeparator) {
    return {
      type: "RepetitionMandatoryWithSeparator",
      idx: node.idx,
      separator: serializeProduction(new Terminal({ terminalType: node.separator })),
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof RepetitionWithSeparator) {
    return {
      type: "RepetitionWithSeparator",
      idx: node.idx,
      separator: serializeProduction(new Terminal({ terminalType: node.separator })),
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof Repetition) {
    return {
      type: "Repetition",
      idx: node.idx,
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof Alternation) {
    return {
      type: "Alternation",
      idx: node.idx,
      definition: convertDefinition(node.definition)
    };
  } else if (node instanceof Terminal) {
    const serializedTerminal = {
      type: "Terminal",
      name: node.terminalType.name,
      label: tokenLabel(node.terminalType),
      idx: node.idx
    };
    if (isString_default(node.label)) {
      serializedTerminal.terminalLabel = node.label;
    }
    const pattern = node.terminalType.PATTERN;
    if (node.terminalType.PATTERN) {
      serializedTerminal.pattern = isRegExp_default(pattern) ? pattern.source : pattern;
    }
    return serializedTerminal;
  } else if (node instanceof Rule) {
    return {
      type: "Rule",
      name: node.name,
      orgText: node.orgText,
      definition: convertDefinition(node.definition)
    };
  } else {
    throw Error("non exhaustive match");
  }
}

// node_modules/@chevrotain/gast/lib/src/visitor.js
var GAstVisitor = class {
  visit(node) {
    const nodeAny = node;
    switch (nodeAny.constructor) {
      case NonTerminal:
        return this.visitNonTerminal(nodeAny);
      case Alternative:
        return this.visitAlternative(nodeAny);
      case Option:
        return this.visitOption(nodeAny);
      case RepetitionMandatory:
        return this.visitRepetitionMandatory(nodeAny);
      case RepetitionMandatoryWithSeparator:
        return this.visitRepetitionMandatoryWithSeparator(nodeAny);
      case RepetitionWithSeparator:
        return this.visitRepetitionWithSeparator(nodeAny);
      case Repetition:
        return this.visitRepetition(nodeAny);
      case Alternation:
        return this.visitAlternation(nodeAny);
      case Terminal:
        return this.visitTerminal(nodeAny);
      case Rule:
        return this.visitRule(nodeAny);
      default:
        throw Error("non exhaustive match");
    }
  }
  /* c8 ignore next */
  visitNonTerminal(node) {
  }
  /* c8 ignore next */
  visitAlternative(node) {
  }
  /* c8 ignore next */
  visitOption(node) {
  }
  /* c8 ignore next */
  visitRepetition(node) {
  }
  /* c8 ignore next */
  visitRepetitionMandatory(node) {
  }
  /* c8 ignore next 3 */
  visitRepetitionMandatoryWithSeparator(node) {
  }
  /* c8 ignore next */
  visitRepetitionWithSeparator(node) {
  }
  /* c8 ignore next */
  visitAlternation(node) {
  }
  /* c8 ignore next */
  visitTerminal(node) {
  }
  /* c8 ignore next */
  visitRule(node) {
  }
};

// node_modules/@chevrotain/gast/lib/src/helpers.js
function isSequenceProd(prod) {
  return prod instanceof Alternative || prod instanceof Option || prod instanceof Repetition || prod instanceof RepetitionMandatory || prod instanceof RepetitionMandatoryWithSeparator || prod instanceof RepetitionWithSeparator || prod instanceof Terminal || prod instanceof Rule;
}
function isOptionalProd(prod, alreadyVisited = []) {
  const isDirectlyOptional = prod instanceof Option || prod instanceof Repetition || prod instanceof RepetitionWithSeparator;
  if (isDirectlyOptional) {
    return true;
  }
  if (prod instanceof Alternation) {
    return some_default(prod.definition, (subProd) => {
      return isOptionalProd(subProd, alreadyVisited);
    });
  } else if (prod instanceof NonTerminal && includes_default(alreadyVisited, prod)) {
    return false;
  } else if (prod instanceof AbstractProduction) {
    if (prod instanceof NonTerminal) {
      alreadyVisited.push(prod);
    }
    return every_default(prod.definition, (subProd) => {
      return isOptionalProd(subProd, alreadyVisited);
    });
  } else {
    return false;
  }
}
function isBranchingProd(prod) {
  return prod instanceof Alternation;
}
function getProductionDslName(prod) {
  if (prod instanceof NonTerminal) {
    return "SUBRULE";
  } else if (prod instanceof Option) {
    return "OPTION";
  } else if (prod instanceof Alternation) {
    return "OR";
  } else if (prod instanceof RepetitionMandatory) {
    return "AT_LEAST_ONE";
  } else if (prod instanceof RepetitionMandatoryWithSeparator) {
    return "AT_LEAST_ONE_SEP";
  } else if (prod instanceof RepetitionWithSeparator) {
    return "MANY_SEP";
  } else if (prod instanceof Repetition) {
    return "MANY";
  } else if (prod instanceof Terminal) {
    return "CONSUME";
  } else {
    throw Error("non exhaustive match");
  }
}

// node_modules/chevrotain/lib/src/parse/grammar/rest.js
var RestWalker = class {
  walk(prod, prevRest = []) {
    forEach_default(prod.definition, (subProd, index) => {
      const currRest = drop_default(prod.definition, index + 1);
      if (subProd instanceof NonTerminal) {
        this.walkProdRef(subProd, currRest, prevRest);
      } else if (subProd instanceof Terminal) {
        this.walkTerminal(subProd, currRest, prevRest);
      } else if (subProd instanceof Alternative) {
        this.walkFlat(subProd, currRest, prevRest);
      } else if (subProd instanceof Option) {
        this.walkOption(subProd, currRest, prevRest);
      } else if (subProd instanceof RepetitionMandatory) {
        this.walkAtLeastOne(subProd, currRest, prevRest);
      } else if (subProd instanceof RepetitionMandatoryWithSeparator) {
        this.walkAtLeastOneSep(subProd, currRest, prevRest);
      } else if (subProd instanceof RepetitionWithSeparator) {
        this.walkManySep(subProd, currRest, prevRest);
      } else if (subProd instanceof Repetition) {
        this.walkMany(subProd, currRest, prevRest);
      } else if (subProd instanceof Alternation) {
        this.walkOr(subProd, currRest, prevRest);
      } else {
        throw Error("non exhaustive match");
      }
    });
  }
  walkTerminal(terminal, currRest, prevRest) {
  }
  walkProdRef(refProd, currRest, prevRest) {
  }
  walkFlat(flatProd, currRest, prevRest) {
    const fullOrRest = currRest.concat(prevRest);
    this.walk(flatProd, fullOrRest);
  }
  walkOption(optionProd, currRest, prevRest) {
    const fullOrRest = currRest.concat(prevRest);
    this.walk(optionProd, fullOrRest);
  }
  walkAtLeastOne(atLeastOneProd, currRest, prevRest) {
    const fullAtLeastOneRest = [
      new Option({ definition: atLeastOneProd.definition })
    ].concat(currRest, prevRest);
    this.walk(atLeastOneProd, fullAtLeastOneRest);
  }
  walkAtLeastOneSep(atLeastOneSepProd, currRest, prevRest) {
    const fullAtLeastOneSepRest = restForRepetitionWithSeparator(atLeastOneSepProd, currRest, prevRest);
    this.walk(atLeastOneSepProd, fullAtLeastOneSepRest);
  }
  walkMany(manyProd, currRest, prevRest) {
    const fullManyRest = [
      new Option({ definition: manyProd.definition })
    ].concat(currRest, prevRest);
    this.walk(manyProd, fullManyRest);
  }
  walkManySep(manySepProd, currRest, prevRest) {
    const fullManySepRest = restForRepetitionWithSeparator(manySepProd, currRest, prevRest);
    this.walk(manySepProd, fullManySepRest);
  }
  walkOr(orProd, currRest, prevRest) {
    const fullOrRest = currRest.concat(prevRest);
    forEach_default(orProd.definition, (alt) => {
      const prodWrapper = new Alternative({ definition: [alt] });
      this.walk(prodWrapper, fullOrRest);
    });
  }
};
function restForRepetitionWithSeparator(repSepProd, currRest, prevRest) {
  const repSepRest = [
    new Option({
      definition: [
        new Terminal({ terminalType: repSepProd.separator })
      ].concat(repSepProd.definition)
    })
  ];
  const fullRepSepRest = repSepRest.concat(currRest, prevRest);
  return fullRepSepRest;
}

// node_modules/chevrotain/lib/src/parse/grammar/first.js
function first(prod) {
  if (prod instanceof NonTerminal) {
    return first(prod.referencedRule);
  } else if (prod instanceof Terminal) {
    return firstForTerminal(prod);
  } else if (isSequenceProd(prod)) {
    return firstForSequence(prod);
  } else if (isBranchingProd(prod)) {
    return firstForBranching(prod);
  } else {
    throw Error("non exhaustive match");
  }
}
function firstForSequence(prod) {
  let firstSet = [];
  const seq = prod.definition;
  let nextSubProdIdx = 0;
  let hasInnerProdsRemaining = seq.length > nextSubProdIdx;
  let currSubProd;
  let isLastInnerProdOptional = true;
  while (hasInnerProdsRemaining && isLastInnerProdOptional) {
    currSubProd = seq[nextSubProdIdx];
    isLastInnerProdOptional = isOptionalProd(currSubProd);
    firstSet = firstSet.concat(first(currSubProd));
    nextSubProdIdx = nextSubProdIdx + 1;
    hasInnerProdsRemaining = seq.length > nextSubProdIdx;
  }
  return uniq_default(firstSet);
}
function firstForBranching(prod) {
  const allAlternativesFirsts = map_default(prod.definition, (innerProd) => {
    return first(innerProd);
  });
  return uniq_default(flatten_default(allAlternativesFirsts));
}
function firstForTerminal(terminal) {
  return [terminal.terminalType];
}

// node_modules/chevrotain/lib/src/parse/constants.js
var IN = "_~IN~_";

// node_modules/chevrotain/lib/src/parse/grammar/follow.js
var ResyncFollowsWalker = class extends RestWalker {
  constructor(topProd) {
    super();
    this.topProd = topProd;
    this.follows = {};
  }
  startWalking() {
    this.walk(this.topProd);
    return this.follows;
  }
  walkTerminal(terminal, currRest, prevRest) {
  }
  walkProdRef(refProd, currRest, prevRest) {
    const followName = buildBetweenProdsFollowPrefix(refProd.referencedRule, refProd.idx) + this.topProd.name;
    const fullRest = currRest.concat(prevRest);
    const restProd = new Alternative({ definition: fullRest });
    const t_in_topProd_follows = first(restProd);
    this.follows[followName] = t_in_topProd_follows;
  }
};
function computeAllProdsFollows(topProductions) {
  const reSyncFollows = {};
  forEach_default(topProductions, (topProd) => {
    const currRefsFollow = new ResyncFollowsWalker(topProd).startWalking();
    assign_default(reSyncFollows, currRefsFollow);
  });
  return reSyncFollows;
}
function buildBetweenProdsFollowPrefix(inner, occurenceInParent) {
  return inner.name + occurenceInParent + IN;
}

// node_modules/chevrotain/lib/src/scan/reg_exp_parser.js
var regExpAstCache = {};
var regExpParser = new RegExpParser();
function getRegExpAst(regExp) {
  const regExpStr = regExp.toString();
  if (regExpAstCache.hasOwnProperty(regExpStr)) {
    return regExpAstCache[regExpStr];
  } else {
    const regExpAst = regExpParser.pattern(regExpStr);
    regExpAstCache[regExpStr] = regExpAst;
    return regExpAst;
  }
}
function clearRegExpParserCache() {
  regExpAstCache = {};
}

// node_modules/chevrotain/lib/src/scan/reg_exp.js
var complementErrorMessage = "Complement Sets are not supported for first char optimization";
var failedOptimizationPrefixMsg = 'Unable to use "first char" lexer optimizations:\n';
function getOptimizedStartCodesIndices(regExp, ensureOptimizations = false) {
  try {
    const ast = getRegExpAst(regExp);
    const firstChars = firstCharOptimizedIndices(ast.value, {}, ast.flags.ignoreCase);
    return firstChars;
  } catch (e) {
    if (e.message === complementErrorMessage) {
      if (ensureOptimizations) {
        PRINT_WARNING(`${failedOptimizationPrefixMsg}	Unable to optimize: < ${regExp.toString()} >
	Complement Sets cannot be automatically optimized.
	This will disable the lexer's first char optimizations.
	See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#COMPLEMENT for details.`);
      }
    } else {
      let msgSuffix = "";
      if (ensureOptimizations) {
        msgSuffix = "\n	This will disable the lexer's first char optimizations.\n	See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#REGEXP_PARSING for details.";
      }
      PRINT_ERROR(`${failedOptimizationPrefixMsg}
	Failed parsing: < ${regExp.toString()} >
	Using the @chevrotain/regexp-to-ast library
	Please open an issue at: https://github.com/chevrotain/chevrotain/issues` + msgSuffix);
    }
  }
  return [];
}
function firstCharOptimizedIndices(ast, result2, ignoreCase) {
  switch (ast.type) {
    case "Disjunction":
      for (let i = 0; i < ast.value.length; i++) {
        firstCharOptimizedIndices(ast.value[i], result2, ignoreCase);
      }
      break;
    case "Alternative":
      const terms = ast.value;
      for (let i = 0; i < terms.length; i++) {
        const term = terms[i];
        switch (term.type) {
          case "EndAnchor":
          case "GroupBackReference":
          case "Lookahead":
          case "NegativeLookahead":
          case "StartAnchor":
          case "WordBoundary":
          case "NonWordBoundary":
            continue;
        }
        const atom2 = term;
        switch (atom2.type) {
          case "Character":
            addOptimizedIdxToResult(atom2.value, result2, ignoreCase);
            break;
          case "Set":
            if (atom2.complement === true) {
              throw Error(complementErrorMessage);
            }
            forEach_default(atom2.value, (code) => {
              if (typeof code === "number") {
                addOptimizedIdxToResult(code, result2, ignoreCase);
              } else {
                const range2 = code;
                if (ignoreCase === true) {
                  for (let rangeCode = range2.from; rangeCode <= range2.to; rangeCode++) {
                    addOptimizedIdxToResult(rangeCode, result2, ignoreCase);
                  }
                } else {
                  for (let rangeCode = range2.from; rangeCode <= range2.to && rangeCode < minOptimizationVal; rangeCode++) {
                    addOptimizedIdxToResult(rangeCode, result2, ignoreCase);
                  }
                  if (range2.to >= minOptimizationVal) {
                    const minUnOptVal = range2.from >= minOptimizationVal ? range2.from : minOptimizationVal;
                    const maxUnOptVal = range2.to;
                    const minOptIdx = charCodeToOptimizedIndex(minUnOptVal);
                    const maxOptIdx = charCodeToOptimizedIndex(maxUnOptVal);
                    for (let currOptIdx = minOptIdx; currOptIdx <= maxOptIdx; currOptIdx++) {
                      result2[currOptIdx] = currOptIdx;
                    }
                  }
                }
              }
            });
            break;
          case "Group":
            firstCharOptimizedIndices(atom2.value, result2, ignoreCase);
            break;
          default:
            throw Error("Non Exhaustive Match");
        }
        const isOptionalQuantifier = atom2.quantifier !== void 0 && atom2.quantifier.atLeast === 0;
        if (
          // A group may be optional due to empty contents /(?:)/
          // or if everything inside it is optional /((a)?)/
          atom2.type === "Group" && isWholeOptional(atom2) === false || // If this term is not a group it may only be optional if it has an optional quantifier
          atom2.type !== "Group" && isOptionalQuantifier === false
        ) {
          break;
        }
      }
      break;
    default:
      throw Error("non exhaustive match!");
  }
  return values_default(result2);
}
function addOptimizedIdxToResult(code, result2, ignoreCase) {
  const optimizedCharIdx = charCodeToOptimizedIndex(code);
  result2[optimizedCharIdx] = optimizedCharIdx;
  if (ignoreCase === true) {
    handleIgnoreCase(code, result2);
  }
}
function handleIgnoreCase(code, result2) {
  const char = String.fromCharCode(code);
  const upperChar = char.toUpperCase();
  if (upperChar !== char) {
    const optimizedCharIdx = charCodeToOptimizedIndex(upperChar.charCodeAt(0));
    result2[optimizedCharIdx] = optimizedCharIdx;
  } else {
    const lowerChar = char.toLowerCase();
    if (lowerChar !== char) {
      const optimizedCharIdx = charCodeToOptimizedIndex(lowerChar.charCodeAt(0));
      result2[optimizedCharIdx] = optimizedCharIdx;
    }
  }
}
function findCode(setNode, targetCharCodes) {
  return find_default(setNode.value, (codeOrRange) => {
    if (typeof codeOrRange === "number") {
      return includes_default(targetCharCodes, codeOrRange);
    } else {
      const range2 = codeOrRange;
      return find_default(targetCharCodes, (targetCode) => range2.from <= targetCode && targetCode <= range2.to) !== void 0;
    }
  });
}
function isWholeOptional(ast) {
  const quantifier = ast.quantifier;
  if (quantifier && quantifier.atLeast === 0) {
    return true;
  }
  if (!ast.value) {
    return false;
  }
  return isArray_default(ast.value) ? every_default(ast.value, isWholeOptional) : isWholeOptional(ast.value);
}
var CharCodeFinder = class extends BaseRegExpVisitor {
  constructor(targetCharCodes) {
    super();
    this.targetCharCodes = targetCharCodes;
    this.found = false;
  }
  visitChildren(node) {
    if (this.found === true) {
      return;
    }
    switch (node.type) {
      case "Lookahead":
        this.visitLookahead(node);
        return;
      case "NegativeLookahead":
        this.visitNegativeLookahead(node);
        return;
    }
    super.visitChildren(node);
  }
  visitCharacter(node) {
    if (includes_default(this.targetCharCodes, node.value)) {
      this.found = true;
    }
  }
  visitSet(node) {
    if (node.complement) {
      if (findCode(node, this.targetCharCodes) === void 0) {
        this.found = true;
      }
    } else {
      if (findCode(node, this.targetCharCodes) !== void 0) {
        this.found = true;
      }
    }
  }
};
function canMatchCharCode(charCodes, pattern) {
  if (pattern instanceof RegExp) {
    const ast = getRegExpAst(pattern);
    const charCodeFinder = new CharCodeFinder(charCodes);
    charCodeFinder.visit(ast);
    return charCodeFinder.found;
  } else {
    return find_default(pattern, (char) => {
      return includes_default(charCodes, char.charCodeAt(0));
    }) !== void 0;
  }
}

// node_modules/chevrotain/lib/src/scan/lexer.js
var PATTERN = "PATTERN";
var DEFAULT_MODE = "defaultMode";
var MODES = "modes";
var SUPPORT_STICKY = typeof new RegExp("(?:)").sticky === "boolean";
function analyzeTokenTypes(tokenTypes, options) {
  options = defaults_default(options, {
    useSticky: SUPPORT_STICKY,
    debug: false,
    safeMode: false,
    positionTracking: "full",
    lineTerminatorCharacters: ["\r", "\n"],
    tracer: (msg, action) => action()
  });
  const tracer = options.tracer;
  tracer("initCharCodeToOptimizedIndexMap", () => {
    initCharCodeToOptimizedIndexMap();
  });
  let onlyRelevantTypes;
  tracer("Reject Lexer.NA", () => {
    onlyRelevantTypes = reject_default(tokenTypes, (currType) => {
      return currType[PATTERN] === Lexer.NA;
    });
  });
  let hasCustom = false;
  let allTransformedPatterns;
  tracer("Transform Patterns", () => {
    hasCustom = false;
    allTransformedPatterns = map_default(onlyRelevantTypes, (currType) => {
      const currPattern = currType[PATTERN];
      if (isRegExp_default(currPattern)) {
        const regExpSource = currPattern.source;
        if (regExpSource.length === 1 && // only these regExp meta characters which can appear in a length one regExp
        regExpSource !== "^" && regExpSource !== "$" && regExpSource !== "." && !currPattern.ignoreCase) {
          return regExpSource;
        } else if (regExpSource.length === 2 && regExpSource[0] === "\\" && // not a meta character
        !includes_default([
          "d",
          "D",
          "s",
          "S",
          "t",
          "r",
          "n",
          "t",
          "0",
          "c",
          "b",
          "B",
          "f",
          "v",
          "w",
          "W"
        ], regExpSource[1])) {
          return regExpSource[1];
        } else {
          return options.useSticky ? addStickyFlag(currPattern) : addStartOfInput(currPattern);
        }
      } else if (isFunction_default(currPattern)) {
        hasCustom = true;
        return { exec: currPattern };
      } else if (typeof currPattern === "object") {
        hasCustom = true;
        return currPattern;
      } else if (typeof currPattern === "string") {
        if (currPattern.length === 1) {
          return currPattern;
        } else {
          const escapedRegExpString = currPattern.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
          const wrappedRegExp = new RegExp(escapedRegExpString);
          return options.useSticky ? addStickyFlag(wrappedRegExp) : addStartOfInput(wrappedRegExp);
        }
      } else {
        throw Error("non exhaustive match");
      }
    });
  });
  let patternIdxToType;
  let patternIdxToGroup;
  let patternIdxToLongerAltIdxArr;
  let patternIdxToPushMode;
  let patternIdxToPopMode;
  tracer("misc mapping", () => {
    patternIdxToType = map_default(onlyRelevantTypes, (currType) => currType.tokenTypeIdx);
    patternIdxToGroup = map_default(onlyRelevantTypes, (clazz) => {
      const groupName = clazz.GROUP;
      if (groupName === Lexer.SKIPPED) {
        return void 0;
      } else if (isString_default(groupName)) {
        return groupName;
      } else if (isUndefined_default(groupName)) {
        return false;
      } else {
        throw Error("non exhaustive match");
      }
    });
    patternIdxToLongerAltIdxArr = map_default(onlyRelevantTypes, (clazz) => {
      const longerAltType = clazz.LONGER_ALT;
      if (longerAltType) {
        const longerAltIdxArr = isArray_default(longerAltType) ? map_default(longerAltType, (type) => indexOf_default(onlyRelevantTypes, type)) : [indexOf_default(onlyRelevantTypes, longerAltType)];
        return longerAltIdxArr;
      }
    });
    patternIdxToPushMode = map_default(onlyRelevantTypes, (clazz) => clazz.PUSH_MODE);
    patternIdxToPopMode = map_default(onlyRelevantTypes, (clazz) => has_default(clazz, "POP_MODE"));
  });
  let patternIdxToCanLineTerminator;
  tracer("Line Terminator Handling", () => {
    const lineTerminatorCharCodes = getCharCodes(options.lineTerminatorCharacters);
    patternIdxToCanLineTerminator = map_default(onlyRelevantTypes, (tokType) => false);
    if (options.positionTracking !== "onlyOffset") {
      patternIdxToCanLineTerminator = map_default(onlyRelevantTypes, (tokType) => {
        if (has_default(tokType, "LINE_BREAKS")) {
          return !!tokType.LINE_BREAKS;
        } else {
          return checkLineBreaksIssues(tokType, lineTerminatorCharCodes) === false && canMatchCharCode(lineTerminatorCharCodes, tokType.PATTERN);
        }
      });
    }
  });
  let patternIdxToIsCustom;
  let patternIdxToShort;
  let emptyGroups;
  let patternIdxToConfig;
  tracer("Misc Mapping #2", () => {
    patternIdxToIsCustom = map_default(onlyRelevantTypes, isCustomPattern);
    patternIdxToShort = map_default(allTransformedPatterns, isShortPattern);
    emptyGroups = reduce_default(onlyRelevantTypes, (acc, clazz) => {
      const groupName = clazz.GROUP;
      if (isString_default(groupName) && !(groupName === Lexer.SKIPPED)) {
        acc[groupName] = [];
      }
      return acc;
    }, {});
    patternIdxToConfig = map_default(allTransformedPatterns, (x, idx) => {
      return {
        pattern: allTransformedPatterns[idx],
        longerAlt: patternIdxToLongerAltIdxArr[idx],
        canLineTerminator: patternIdxToCanLineTerminator[idx],
        isCustom: patternIdxToIsCustom[idx],
        short: patternIdxToShort[idx],
        group: patternIdxToGroup[idx],
        push: patternIdxToPushMode[idx],
        pop: patternIdxToPopMode[idx],
        tokenTypeIdx: patternIdxToType[idx],
        tokenType: onlyRelevantTypes[idx]
      };
    });
  });
  let canBeOptimized = true;
  let charCodeToPatternIdxToConfig = [];
  if (!options.safeMode) {
    tracer("First Char Optimization", () => {
      charCodeToPatternIdxToConfig = reduce_default(onlyRelevantTypes, (result2, currTokType, idx) => {
        if (typeof currTokType.PATTERN === "string") {
          const charCode = currTokType.PATTERN.charCodeAt(0);
          const optimizedIdx = charCodeToOptimizedIndex(charCode);
          addToMapOfArrays(result2, optimizedIdx, patternIdxToConfig[idx]);
        } else if (isArray_default(currTokType.START_CHARS_HINT)) {
          let lastOptimizedIdx;
          forEach_default(currTokType.START_CHARS_HINT, (charOrInt) => {
            const charCode = typeof charOrInt === "string" ? charOrInt.charCodeAt(0) : charOrInt;
            const currOptimizedIdx = charCodeToOptimizedIndex(charCode);
            if (lastOptimizedIdx !== currOptimizedIdx) {
              lastOptimizedIdx = currOptimizedIdx;
              addToMapOfArrays(result2, currOptimizedIdx, patternIdxToConfig[idx]);
            }
          });
        } else if (isRegExp_default(currTokType.PATTERN)) {
          if (currTokType.PATTERN.unicode) {
            canBeOptimized = false;
            if (options.ensureOptimizations) {
              PRINT_ERROR(`${failedOptimizationPrefixMsg}	Unable to analyze < ${currTokType.PATTERN.toString()} > pattern.
	The regexp unicode flag is not currently supported by the regexp-to-ast library.
	This will disable the lexer's first char optimizations.
	For details See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#UNICODE_OPTIMIZE`);
            }
          } else {
            const optimizedCodes = getOptimizedStartCodesIndices(currTokType.PATTERN, options.ensureOptimizations);
            if (isEmpty_default(optimizedCodes)) {
              canBeOptimized = false;
            }
            forEach_default(optimizedCodes, (code) => {
              addToMapOfArrays(result2, code, patternIdxToConfig[idx]);
            });
          }
        } else {
          if (options.ensureOptimizations) {
            PRINT_ERROR(`${failedOptimizationPrefixMsg}	TokenType: <${currTokType.name}> is using a custom token pattern without providing <start_chars_hint> parameter.
	This will disable the lexer's first char optimizations.
	For details See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#CUSTOM_OPTIMIZE`);
          }
          canBeOptimized = false;
        }
        return result2;
      }, []);
    });
  }
  return {
    emptyGroups,
    patternIdxToConfig,
    charCodeToPatternIdxToConfig,
    hasCustom,
    canBeOptimized
  };
}
function validatePatterns(tokenTypes, validModesNames) {
  let errors = [];
  const missingResult = findMissingPatterns(tokenTypes);
  errors = errors.concat(missingResult.errors);
  const invalidResult = findInvalidPatterns(missingResult.valid);
  const validTokenTypes = invalidResult.valid;
  errors = errors.concat(invalidResult.errors);
  errors = errors.concat(validateRegExpPattern(validTokenTypes));
  errors = errors.concat(findInvalidGroupType(validTokenTypes));
  errors = errors.concat(findModesThatDoNotExist(validTokenTypes, validModesNames));
  errors = errors.concat(findUnreachablePatterns(validTokenTypes));
  return errors;
}
function validateRegExpPattern(tokenTypes) {
  let errors = [];
  const withRegExpPatterns = filter_default(tokenTypes, (currTokType) => isRegExp_default(currTokType[PATTERN]));
  errors = errors.concat(findEndOfInputAnchor(withRegExpPatterns));
  errors = errors.concat(findStartOfInputAnchor(withRegExpPatterns));
  errors = errors.concat(findUnsupportedFlags(withRegExpPatterns));
  errors = errors.concat(findDuplicatePatterns(withRegExpPatterns));
  errors = errors.concat(findEmptyMatchRegExps(withRegExpPatterns));
  return errors;
}
function findMissingPatterns(tokenTypes) {
  const tokenTypesWithMissingPattern = filter_default(tokenTypes, (currType) => {
    return !has_default(currType, PATTERN);
  });
  const errors = map_default(tokenTypesWithMissingPattern, (currType) => {
    return {
      message: "Token Type: ->" + currType.name + "<- missing static 'PATTERN' property",
      type: LexerDefinitionErrorType.MISSING_PATTERN,
      tokenTypes: [currType]
    };
  });
  const valid = difference_default(tokenTypes, tokenTypesWithMissingPattern);
  return { errors, valid };
}
function findInvalidPatterns(tokenTypes) {
  const tokenTypesWithInvalidPattern = filter_default(tokenTypes, (currType) => {
    const pattern = currType[PATTERN];
    return !isRegExp_default(pattern) && !isFunction_default(pattern) && !has_default(pattern, "exec") && !isString_default(pattern);
  });
  const errors = map_default(tokenTypesWithInvalidPattern, (currType) => {
    return {
      message: "Token Type: ->" + currType.name + "<- static 'PATTERN' can only be a RegExp, a Function matching the {CustomPatternMatcherFunc} type or an Object matching the {ICustomPattern} interface.",
      type: LexerDefinitionErrorType.INVALID_PATTERN,
      tokenTypes: [currType]
    };
  });
  const valid = difference_default(tokenTypes, tokenTypesWithInvalidPattern);
  return { errors, valid };
}
var end_of_input = /[^\\][$]/;
function findEndOfInputAnchor(tokenTypes) {
  class EndAnchorFinder extends BaseRegExpVisitor {
    constructor() {
      super(...arguments);
      this.found = false;
    }
    visitEndAnchor(node) {
      this.found = true;
    }
  }
  const invalidRegex = filter_default(tokenTypes, (currType) => {
    const pattern = currType.PATTERN;
    try {
      const regexpAst = getRegExpAst(pattern);
      const endAnchorVisitor = new EndAnchorFinder();
      endAnchorVisitor.visit(regexpAst);
      return endAnchorVisitor.found;
    } catch (e) {
      return end_of_input.test(pattern.source);
    }
  });
  const errors = map_default(invalidRegex, (currType) => {
    return {
      message: "Unexpected RegExp Anchor Error:\n	Token Type: ->" + currType.name + "<- static 'PATTERN' cannot contain end of input anchor '$'\n	See chevrotain.io/docs/guide/resolving_lexer_errors.html#ANCHORS	for details.",
      type: LexerDefinitionErrorType.EOI_ANCHOR_FOUND,
      tokenTypes: [currType]
    };
  });
  return errors;
}
function findEmptyMatchRegExps(tokenTypes) {
  const matchesEmptyString = filter_default(tokenTypes, (currType) => {
    const pattern = currType.PATTERN;
    return pattern.test("");
  });
  const errors = map_default(matchesEmptyString, (currType) => {
    return {
      message: "Token Type: ->" + currType.name + "<- static 'PATTERN' must not match an empty string",
      type: LexerDefinitionErrorType.EMPTY_MATCH_PATTERN,
      tokenTypes: [currType]
    };
  });
  return errors;
}
var start_of_input = /[^\\[][\^]|^\^/;
function findStartOfInputAnchor(tokenTypes) {
  class StartAnchorFinder extends BaseRegExpVisitor {
    constructor() {
      super(...arguments);
      this.found = false;
    }
    visitStartAnchor(node) {
      this.found = true;
    }
  }
  const invalidRegex = filter_default(tokenTypes, (currType) => {
    const pattern = currType.PATTERN;
    try {
      const regexpAst = getRegExpAst(pattern);
      const startAnchorVisitor = new StartAnchorFinder();
      startAnchorVisitor.visit(regexpAst);
      return startAnchorVisitor.found;
    } catch (e) {
      return start_of_input.test(pattern.source);
    }
  });
  const errors = map_default(invalidRegex, (currType) => {
    return {
      message: "Unexpected RegExp Anchor Error:\n	Token Type: ->" + currType.name + "<- static 'PATTERN' cannot contain start of input anchor '^'\n	See https://chevrotain.io/docs/guide/resolving_lexer_errors.html#ANCHORS	for details.",
      type: LexerDefinitionErrorType.SOI_ANCHOR_FOUND,
      tokenTypes: [currType]
    };
  });
  return errors;
}
function findUnsupportedFlags(tokenTypes) {
  const invalidFlags = filter_default(tokenTypes, (currType) => {
    const pattern = currType[PATTERN];
    return pattern instanceof RegExp && (pattern.multiline || pattern.global);
  });
  const errors = map_default(invalidFlags, (currType) => {
    return {
      message: "Token Type: ->" + currType.name + "<- static 'PATTERN' may NOT contain global('g') or multiline('m')",
      type: LexerDefinitionErrorType.UNSUPPORTED_FLAGS_FOUND,
      tokenTypes: [currType]
    };
  });
  return errors;
}
function findDuplicatePatterns(tokenTypes) {
  const found = [];
  let identicalPatterns = map_default(tokenTypes, (outerType) => {
    return reduce_default(tokenTypes, (result2, innerType) => {
      if (outerType.PATTERN.source === innerType.PATTERN.source && !includes_default(found, innerType) && innerType.PATTERN !== Lexer.NA) {
        found.push(innerType);
        result2.push(innerType);
        return result2;
      }
      return result2;
    }, []);
  });
  identicalPatterns = compact_default(identicalPatterns);
  const duplicatePatterns = filter_default(identicalPatterns, (currIdenticalSet) => {
    return currIdenticalSet.length > 1;
  });
  const errors = map_default(duplicatePatterns, (setOfIdentical) => {
    const tokenTypeNames = map_default(setOfIdentical, (currType) => {
      return currType.name;
    });
    const dupPatternSrc = head_default(setOfIdentical).PATTERN;
    return {
      message: `The same RegExp pattern ->${dupPatternSrc}<-has been used in all of the following Token Types: ${tokenTypeNames.join(", ")} <-`,
      type: LexerDefinitionErrorType.DUPLICATE_PATTERNS_FOUND,
      tokenTypes: setOfIdentical
    };
  });
  return errors;
}
function findInvalidGroupType(tokenTypes) {
  const invalidTypes = filter_default(tokenTypes, (clazz) => {
    if (!has_default(clazz, "GROUP")) {
      return false;
    }
    const group = clazz.GROUP;
    return group !== Lexer.SKIPPED && group !== Lexer.NA && !isString_default(group);
  });
  const errors = map_default(invalidTypes, (currType) => {
    return {
      message: "Token Type: ->" + currType.name + "<- static 'GROUP' can only be Lexer.SKIPPED/Lexer.NA/A String",
      type: LexerDefinitionErrorType.INVALID_GROUP_TYPE_FOUND,
      tokenTypes: [currType]
    };
  });
  return errors;
}
function findModesThatDoNotExist(tokenTypes, validModes) {
  const invalidModes = filter_default(tokenTypes, (clazz) => {
    return clazz.PUSH_MODE !== void 0 && !includes_default(validModes, clazz.PUSH_MODE);
  });
  const errors = map_default(invalidModes, (tokType) => {
    const msg = `Token Type: ->${tokType.name}<- static 'PUSH_MODE' value cannot refer to a Lexer Mode ->${tokType.PUSH_MODE}<-which does not exist`;
    return {
      message: msg,
      type: LexerDefinitionErrorType.PUSH_MODE_DOES_NOT_EXIST,
      tokenTypes: [tokType]
    };
  });
  return errors;
}
function findUnreachablePatterns(tokenTypes) {
  const errors = [];
  const canBeTested = reduce_default(tokenTypes, (result2, tokType, idx) => {
    const pattern = tokType.PATTERN;
    if (pattern === Lexer.NA) {
      return result2;
    }
    if (isString_default(pattern)) {
      result2.push({ str: pattern, idx, tokenType: tokType });
    } else if (isRegExp_default(pattern) && noMetaChar(pattern)) {
      result2.push({ str: pattern.source, idx, tokenType: tokType });
    }
    return result2;
  }, []);
  forEach_default(tokenTypes, (tokType, testIdx) => {
    forEach_default(canBeTested, ({ str, idx, tokenType }) => {
      if (testIdx < idx && testTokenType(str, tokType.PATTERN)) {
        const msg = `Token: ->${tokenType.name}<- can never be matched.
Because it appears AFTER the Token Type ->${tokType.name}<-in the lexer's definition.
See https://chevrotain.io/docs/guide/resolving_lexer_errors.html#UNREACHABLE`;
        errors.push({
          message: msg,
          type: LexerDefinitionErrorType.UNREACHABLE_PATTERN,
          tokenTypes: [tokType, tokenType]
        });
      }
    });
  });
  return errors;
}
function testTokenType(str, pattern) {
  if (isRegExp_default(pattern)) {
    const regExpArray = pattern.exec(str);
    return regExpArray !== null && regExpArray.index === 0;
  } else if (isFunction_default(pattern)) {
    return pattern(str, 0, [], {});
  } else if (has_default(pattern, "exec")) {
    return pattern.exec(str, 0, [], {});
  } else if (typeof pattern === "string") {
    return pattern === str;
  } else {
    throw Error("non exhaustive match");
  }
}
function noMetaChar(regExp) {
  const metaChars = [
    ".",
    "\\",
    "[",
    "]",
    "|",
    "^",
    "$",
    "(",
    ")",
    "?",
    "*",
    "+",
    "{"
  ];
  return find_default(metaChars, (char) => regExp.source.indexOf(char) !== -1) === void 0;
}
function addStartOfInput(pattern) {
  const flags = pattern.ignoreCase ? "i" : "";
  return new RegExp(`^(?:${pattern.source})`, flags);
}
function addStickyFlag(pattern) {
  const flags = pattern.ignoreCase ? "iy" : "y";
  return new RegExp(`${pattern.source}`, flags);
}
function performRuntimeChecks(lexerDefinition, trackLines, lineTerminatorCharacters) {
  const errors = [];
  if (!has_default(lexerDefinition, DEFAULT_MODE)) {
    errors.push({
      message: "A MultiMode Lexer cannot be initialized without a <" + DEFAULT_MODE + "> property in its definition\n",
      type: LexerDefinitionErrorType.MULTI_MODE_LEXER_WITHOUT_DEFAULT_MODE
    });
  }
  if (!has_default(lexerDefinition, MODES)) {
    errors.push({
      message: "A MultiMode Lexer cannot be initialized without a <" + MODES + "> property in its definition\n",
      type: LexerDefinitionErrorType.MULTI_MODE_LEXER_WITHOUT_MODES_PROPERTY
    });
  }
  if (has_default(lexerDefinition, MODES) && has_default(lexerDefinition, DEFAULT_MODE) && !has_default(lexerDefinition.modes, lexerDefinition.defaultMode)) {
    errors.push({
      message: `A MultiMode Lexer cannot be initialized with a ${DEFAULT_MODE}: <${lexerDefinition.defaultMode}>which does not exist
`,
      type: LexerDefinitionErrorType.MULTI_MODE_LEXER_DEFAULT_MODE_VALUE_DOES_NOT_EXIST
    });
  }
  if (has_default(lexerDefinition, MODES)) {
    forEach_default(lexerDefinition.modes, (currModeValue, currModeName) => {
      forEach_default(currModeValue, (currTokType, currIdx) => {
        if (isUndefined_default(currTokType)) {
          errors.push({
            message: `A Lexer cannot be initialized using an undefined Token Type. Mode:<${currModeName}> at index: <${currIdx}>
`,
            type: LexerDefinitionErrorType.LEXER_DEFINITION_CANNOT_CONTAIN_UNDEFINED
          });
        } else if (has_default(currTokType, "LONGER_ALT")) {
          const longerAlt = isArray_default(currTokType.LONGER_ALT) ? currTokType.LONGER_ALT : [currTokType.LONGER_ALT];
          forEach_default(longerAlt, (currLongerAlt) => {
            if (!isUndefined_default(currLongerAlt) && !includes_default(currModeValue, currLongerAlt)) {
              errors.push({
                message: `A MultiMode Lexer cannot be initialized with a longer_alt <${currLongerAlt.name}> on token <${currTokType.name}> outside of mode <${currModeName}>
`,
                type: LexerDefinitionErrorType.MULTI_MODE_LEXER_LONGER_ALT_NOT_IN_CURRENT_MODE
              });
            }
          });
        }
      });
    });
  }
  return errors;
}
function performWarningRuntimeChecks(lexerDefinition, trackLines, lineTerminatorCharacters) {
  const warnings = [];
  let hasAnyLineBreak = false;
  const allTokenTypes = compact_default(flatten_default(values_default(lexerDefinition.modes)));
  const concreteTokenTypes = reject_default(allTokenTypes, (currType) => currType[PATTERN] === Lexer.NA);
  const terminatorCharCodes = getCharCodes(lineTerminatorCharacters);
  if (trackLines) {
    forEach_default(concreteTokenTypes, (tokType) => {
      const currIssue = checkLineBreaksIssues(tokType, terminatorCharCodes);
      if (currIssue !== false) {
        const message = buildLineBreakIssueMessage(tokType, currIssue);
        const warningDescriptor = {
          message,
          type: currIssue.issue,
          tokenType: tokType
        };
        warnings.push(warningDescriptor);
      } else {
        if (has_default(tokType, "LINE_BREAKS")) {
          if (tokType.LINE_BREAKS === true) {
            hasAnyLineBreak = true;
          }
        } else {
          if (canMatchCharCode(terminatorCharCodes, tokType.PATTERN)) {
            hasAnyLineBreak = true;
          }
        }
      }
    });
  }
  if (trackLines && !hasAnyLineBreak) {
    warnings.push({
      message: "Warning: No LINE_BREAKS Found.\n	This Lexer has been defined to track line and column information,\n	But none of the Token Types can be identified as matching a line terminator.\n	See https://chevrotain.io/docs/guide/resolving_lexer_errors.html#LINE_BREAKS \n	for details.",
      type: LexerDefinitionErrorType.NO_LINE_BREAKS_FLAGS
    });
  }
  return warnings;
}
function cloneEmptyGroups(emptyGroups) {
  const clonedResult = {};
  const groupKeys = keys_default(emptyGroups);
  forEach_default(groupKeys, (currKey) => {
    const currGroupValue = emptyGroups[currKey];
    if (isArray_default(currGroupValue)) {
      clonedResult[currKey] = [];
    } else {
      throw Error("non exhaustive match");
    }
  });
  return clonedResult;
}
function isCustomPattern(tokenType) {
  const pattern = tokenType.PATTERN;
  if (isRegExp_default(pattern)) {
    return false;
  } else if (isFunction_default(pattern)) {
    return true;
  } else if (has_default(pattern, "exec")) {
    return true;
  } else if (isString_default(pattern)) {
    return false;
  } else {
    throw Error("non exhaustive match");
  }
}
function isShortPattern(pattern) {
  if (isString_default(pattern) && pattern.length === 1) {
    return pattern.charCodeAt(0);
  } else {
    return false;
  }
}
var LineTerminatorOptimizedTester = {
  // implements /\n|\r\n?/g.test
  test: function(text) {
    const len = text.length;
    for (let i = this.lastIndex; i < len; i++) {
      const c = text.charCodeAt(i);
      if (c === 10) {
        this.lastIndex = i + 1;
        return true;
      } else if (c === 13) {
        if (text.charCodeAt(i + 1) === 10) {
          this.lastIndex = i + 2;
        } else {
          this.lastIndex = i + 1;
        }
        return true;
      }
    }
    return false;
  },
  lastIndex: 0
};
function checkLineBreaksIssues(tokType, lineTerminatorCharCodes) {
  if (has_default(tokType, "LINE_BREAKS")) {
    return false;
  } else {
    if (isRegExp_default(tokType.PATTERN)) {
      try {
        canMatchCharCode(lineTerminatorCharCodes, tokType.PATTERN);
      } catch (e) {
        return {
          issue: LexerDefinitionErrorType.IDENTIFY_TERMINATOR,
          errMsg: e.message
        };
      }
      return false;
    } else if (isString_default(tokType.PATTERN)) {
      return false;
    } else if (isCustomPattern(tokType)) {
      return { issue: LexerDefinitionErrorType.CUSTOM_LINE_BREAK };
    } else {
      throw Error("non exhaustive match");
    }
  }
}
function buildLineBreakIssueMessage(tokType, details) {
  if (details.issue === LexerDefinitionErrorType.IDENTIFY_TERMINATOR) {
    return `Warning: unable to identify line terminator usage in pattern.
	The problem is in the <${tokType.name}> Token Type
	 Root cause: ${details.errMsg}.
	For details See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#IDENTIFY_TERMINATOR`;
  } else if (details.issue === LexerDefinitionErrorType.CUSTOM_LINE_BREAK) {
    return `Warning: A Custom Token Pattern should specify the <line_breaks> option.
	The problem is in the <${tokType.name}> Token Type
	For details See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#CUSTOM_LINE_BREAK`;
  } else {
    throw Error("non exhaustive match");
  }
}
function getCharCodes(charsOrCodes) {
  const charCodes = map_default(charsOrCodes, (numOrString) => {
    if (isString_default(numOrString)) {
      return numOrString.charCodeAt(0);
    } else {
      return numOrString;
    }
  });
  return charCodes;
}
function addToMapOfArrays(map2, key, value) {
  if (map2[key] === void 0) {
    map2[key] = [value];
  } else {
    map2[key].push(value);
  }
}
var minOptimizationVal = 256;
var charCodeToOptimizedIdxMap = [];
function charCodeToOptimizedIndex(charCode) {
  return charCode < minOptimizationVal ? charCode : charCodeToOptimizedIdxMap[charCode];
}
function initCharCodeToOptimizedIndexMap() {
  if (isEmpty_default(charCodeToOptimizedIdxMap)) {
    charCodeToOptimizedIdxMap = new Array(65536);
    for (let i = 0; i < 65536; i++) {
      charCodeToOptimizedIdxMap[i] = i > 255 ? 255 + ~~(i / 255) : i;
    }
  }
}

// node_modules/chevrotain/lib/src/scan/tokens.js
function tokenStructuredMatcher(tokInstance, tokConstructor) {
  const instanceType = tokInstance.tokenTypeIdx;
  if (instanceType === tokConstructor.tokenTypeIdx) {
    return true;
  } else {
    return tokConstructor.isParent === true && tokConstructor.categoryMatchesMap[instanceType] === true;
  }
}
function tokenStructuredMatcherNoCategories(token, tokType) {
  return token.tokenTypeIdx === tokType.tokenTypeIdx;
}
var tokenShortNameIdx = 1;
var tokenIdxToClass = {};
function augmentTokenTypes(tokenTypes) {
  const tokenTypesAndParents = expandCategories(tokenTypes);
  assignTokenDefaultProps(tokenTypesAndParents);
  assignCategoriesMapProp(tokenTypesAndParents);
  assignCategoriesTokensProp(tokenTypesAndParents);
  forEach_default(tokenTypesAndParents, (tokType) => {
    tokType.isParent = tokType.categoryMatches.length > 0;
  });
}
function expandCategories(tokenTypes) {
  let result2 = clone_default(tokenTypes);
  let categories = tokenTypes;
  let searching = true;
  while (searching) {
    categories = compact_default(flatten_default(map_default(categories, (currTokType) => currTokType.CATEGORIES)));
    const newCategories = difference_default(categories, result2);
    result2 = result2.concat(newCategories);
    if (isEmpty_default(newCategories)) {
      searching = false;
    } else {
      categories = newCategories;
    }
  }
  return result2;
}
function assignTokenDefaultProps(tokenTypes) {
  forEach_default(tokenTypes, (currTokType) => {
    if (!hasShortKeyProperty(currTokType)) {
      tokenIdxToClass[tokenShortNameIdx] = currTokType;
      currTokType.tokenTypeIdx = tokenShortNameIdx++;
    }
    if (hasCategoriesProperty(currTokType) && !isArray_default(currTokType.CATEGORIES)) {
      currTokType.CATEGORIES = [currTokType.CATEGORIES];
    }
    if (!hasCategoriesProperty(currTokType)) {
      currTokType.CATEGORIES = [];
    }
    if (!hasExtendingTokensTypesProperty(currTokType)) {
      currTokType.categoryMatches = [];
    }
    if (!hasExtendingTokensTypesMapProperty(currTokType)) {
      currTokType.categoryMatchesMap = {};
    }
  });
}
function assignCategoriesTokensProp(tokenTypes) {
  forEach_default(tokenTypes, (currTokType) => {
    currTokType.categoryMatches = [];
    forEach_default(currTokType.categoryMatchesMap, (val, key) => {
      currTokType.categoryMatches.push(tokenIdxToClass[key].tokenTypeIdx);
    });
  });
}
function assignCategoriesMapProp(tokenTypes) {
  forEach_default(tokenTypes, (currTokType) => {
    singleAssignCategoriesToksMap([], currTokType);
  });
}
function singleAssignCategoriesToksMap(path, nextNode) {
  forEach_default(path, (pathNode) => {
    nextNode.categoryMatchesMap[pathNode.tokenTypeIdx] = true;
  });
  forEach_default(nextNode.CATEGORIES, (nextCategory) => {
    const newPath = path.concat(nextNode);
    if (!includes_default(newPath, nextCategory)) {
      singleAssignCategoriesToksMap(newPath, nextCategory);
    }
  });
}
function hasShortKeyProperty(tokType) {
  return has_default(tokType, "tokenTypeIdx");
}
function hasCategoriesProperty(tokType) {
  return has_default(tokType, "CATEGORIES");
}
function hasExtendingTokensTypesProperty(tokType) {
  return has_default(tokType, "categoryMatches");
}
function hasExtendingTokensTypesMapProperty(tokType) {
  return has_default(tokType, "categoryMatchesMap");
}
function isTokenType(tokType) {
  return has_default(tokType, "tokenTypeIdx");
}

// node_modules/chevrotain/lib/src/scan/lexer_errors_public.js
var defaultLexerErrorProvider = {
  buildUnableToPopLexerModeMessage(token) {
    return `Unable to pop Lexer Mode after encountering Token ->${token.image}<- The Mode Stack is empty`;
  },
  buildUnexpectedCharactersMessage(fullText, startOffset, length, line, column) {
    return `unexpected character: ->${fullText.charAt(startOffset)}<- at offset: ${startOffset}, skipped ${length} characters.`;
  }
};

// node_modules/chevrotain/lib/src/scan/lexer_public.js
var LexerDefinitionErrorType;
(function(LexerDefinitionErrorType2) {
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["MISSING_PATTERN"] = 0] = "MISSING_PATTERN";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["INVALID_PATTERN"] = 1] = "INVALID_PATTERN";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["EOI_ANCHOR_FOUND"] = 2] = "EOI_ANCHOR_FOUND";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["UNSUPPORTED_FLAGS_FOUND"] = 3] = "UNSUPPORTED_FLAGS_FOUND";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["DUPLICATE_PATTERNS_FOUND"] = 4] = "DUPLICATE_PATTERNS_FOUND";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["INVALID_GROUP_TYPE_FOUND"] = 5] = "INVALID_GROUP_TYPE_FOUND";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["PUSH_MODE_DOES_NOT_EXIST"] = 6] = "PUSH_MODE_DOES_NOT_EXIST";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["MULTI_MODE_LEXER_WITHOUT_DEFAULT_MODE"] = 7] = "MULTI_MODE_LEXER_WITHOUT_DEFAULT_MODE";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["MULTI_MODE_LEXER_WITHOUT_MODES_PROPERTY"] = 8] = "MULTI_MODE_LEXER_WITHOUT_MODES_PROPERTY";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["MULTI_MODE_LEXER_DEFAULT_MODE_VALUE_DOES_NOT_EXIST"] = 9] = "MULTI_MODE_LEXER_DEFAULT_MODE_VALUE_DOES_NOT_EXIST";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["LEXER_DEFINITION_CANNOT_CONTAIN_UNDEFINED"] = 10] = "LEXER_DEFINITION_CANNOT_CONTAIN_UNDEFINED";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["SOI_ANCHOR_FOUND"] = 11] = "SOI_ANCHOR_FOUND";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["EMPTY_MATCH_PATTERN"] = 12] = "EMPTY_MATCH_PATTERN";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["NO_LINE_BREAKS_FLAGS"] = 13] = "NO_LINE_BREAKS_FLAGS";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["UNREACHABLE_PATTERN"] = 14] = "UNREACHABLE_PATTERN";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["IDENTIFY_TERMINATOR"] = 15] = "IDENTIFY_TERMINATOR";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["CUSTOM_LINE_BREAK"] = 16] = "CUSTOM_LINE_BREAK";
  LexerDefinitionErrorType2[LexerDefinitionErrorType2["MULTI_MODE_LEXER_LONGER_ALT_NOT_IN_CURRENT_MODE"] = 17] = "MULTI_MODE_LEXER_LONGER_ALT_NOT_IN_CURRENT_MODE";
})(LexerDefinitionErrorType || (LexerDefinitionErrorType = {}));
var DEFAULT_LEXER_CONFIG = {
  deferDefinitionErrorsHandling: false,
  positionTracking: "full",
  lineTerminatorsPattern: /\n|\r\n?/g,
  lineTerminatorCharacters: ["\n", "\r"],
  ensureOptimizations: false,
  safeMode: false,
  errorMessageProvider: defaultLexerErrorProvider,
  traceInitPerf: false,
  skipValidations: false,
  recoveryEnabled: true
};
Object.freeze(DEFAULT_LEXER_CONFIG);
var Lexer = class {
  constructor(lexerDefinition, config = DEFAULT_LEXER_CONFIG) {
    this.lexerDefinition = lexerDefinition;
    this.lexerDefinitionErrors = [];
    this.lexerDefinitionWarning = [];
    this.patternIdxToConfig = {};
    this.charCodeToPatternIdxToConfig = {};
    this.modes = [];
    this.emptyGroups = {};
    this.trackStartLines = true;
    this.trackEndLines = true;
    this.hasCustom = false;
    this.canModeBeOptimized = {};
    this.TRACE_INIT = (phaseDesc, phaseImpl) => {
      if (this.traceInitPerf === true) {
        this.traceInitIndent++;
        const indent = new Array(this.traceInitIndent + 1).join("	");
        if (this.traceInitIndent < this.traceInitMaxIdent) {
          console.log(`${indent}--> <${phaseDesc}>`);
        }
        const { time, value } = timer(phaseImpl);
        const traceMethod = time > 10 ? console.warn : console.log;
        if (this.traceInitIndent < this.traceInitMaxIdent) {
          traceMethod(`${indent}<-- <${phaseDesc}> time: ${time}ms`);
        }
        this.traceInitIndent--;
        return value;
      } else {
        return phaseImpl();
      }
    };
    if (typeof config === "boolean") {
      throw Error("The second argument to the Lexer constructor is now an ILexerConfig Object.\na boolean 2nd argument is no longer supported");
    }
    this.config = assign_default({}, DEFAULT_LEXER_CONFIG, config);
    const traceInitVal = this.config.traceInitPerf;
    if (traceInitVal === true) {
      this.traceInitMaxIdent = Infinity;
      this.traceInitPerf = true;
    } else if (typeof traceInitVal === "number") {
      this.traceInitMaxIdent = traceInitVal;
      this.traceInitPerf = true;
    }
    this.traceInitIndent = -1;
    this.TRACE_INIT("Lexer Constructor", () => {
      let actualDefinition;
      let hasOnlySingleMode = true;
      this.TRACE_INIT("Lexer Config handling", () => {
        if (this.config.lineTerminatorsPattern === DEFAULT_LEXER_CONFIG.lineTerminatorsPattern) {
          this.config.lineTerminatorsPattern = LineTerminatorOptimizedTester;
        } else {
          if (this.config.lineTerminatorCharacters === DEFAULT_LEXER_CONFIG.lineTerminatorCharacters) {
            throw Error("Error: Missing <lineTerminatorCharacters> property on the Lexer config.\n	For details See: https://chevrotain.io/docs/guide/resolving_lexer_errors.html#MISSING_LINE_TERM_CHARS");
          }
        }
        if (config.safeMode && config.ensureOptimizations) {
          throw Error('"safeMode" and "ensureOptimizations" flags are mutually exclusive.');
        }
        this.trackStartLines = /full|onlyStart/i.test(this.config.positionTracking);
        this.trackEndLines = /full/i.test(this.config.positionTracking);
        if (isArray_default(lexerDefinition)) {
          actualDefinition = {
            modes: { defaultMode: clone_default(lexerDefinition) },
            defaultMode: DEFAULT_MODE
          };
        } else {
          hasOnlySingleMode = false;
          actualDefinition = clone_default(lexerDefinition);
        }
      });
      if (this.config.skipValidations === false) {
        this.TRACE_INIT("performRuntimeChecks", () => {
          this.lexerDefinitionErrors = this.lexerDefinitionErrors.concat(performRuntimeChecks(actualDefinition, this.trackStartLines, this.config.lineTerminatorCharacters));
        });
        this.TRACE_INIT("performWarningRuntimeChecks", () => {
          this.lexerDefinitionWarning = this.lexerDefinitionWarning.concat(performWarningRuntimeChecks(actualDefinition, this.trackStartLines, this.config.lineTerminatorCharacters));
        });
      }
      actualDefinition.modes = actualDefinition.modes ? actualDefinition.modes : {};
      forEach_default(actualDefinition.modes, (currModeValue, currModeName) => {
        actualDefinition.modes[currModeName] = reject_default(currModeValue, (currTokType) => isUndefined_default(currTokType));
      });
      const allModeNames = keys_default(actualDefinition.modes);
      forEach_default(actualDefinition.modes, (currModDef, currModName) => {
        this.TRACE_INIT(`Mode: <${currModName}> processing`, () => {
          this.modes.push(currModName);
          if (this.config.skipValidations === false) {
            this.TRACE_INIT(`validatePatterns`, () => {
              this.lexerDefinitionErrors = this.lexerDefinitionErrors.concat(validatePatterns(currModDef, allModeNames));
            });
          }
          if (isEmpty_default(this.lexerDefinitionErrors)) {
            augmentTokenTypes(currModDef);
            let currAnalyzeResult;
            this.TRACE_INIT(`analyzeTokenTypes`, () => {
              currAnalyzeResult = analyzeTokenTypes(currModDef, {
                lineTerminatorCharacters: this.config.lineTerminatorCharacters,
                positionTracking: config.positionTracking,
                ensureOptimizations: config.ensureOptimizations,
                safeMode: config.safeMode,
                tracer: this.TRACE_INIT
              });
            });
            this.patternIdxToConfig[currModName] = currAnalyzeResult.patternIdxToConfig;
            this.charCodeToPatternIdxToConfig[currModName] = currAnalyzeResult.charCodeToPatternIdxToConfig;
            this.emptyGroups = assign_default({}, this.emptyGroups, currAnalyzeResult.emptyGroups);
            this.hasCustom = currAnalyzeResult.hasCustom || this.hasCustom;
            this.canModeBeOptimized[currModName] = currAnalyzeResult.canBeOptimized;
          }
        });
      });
      this.defaultMode = actualDefinition.defaultMode;
      if (!isEmpty_default(this.lexerDefinitionErrors) && !this.config.deferDefinitionErrorsHandling) {
        const allErrMessages = map_default(this.lexerDefinitionErrors, (error) => {
          return error.message;
        });
        const allErrMessagesString = allErrMessages.join("-----------------------\n");
        throw new Error("Errors detected in definition of Lexer:\n" + allErrMessagesString);
      }
      forEach_default(this.lexerDefinitionWarning, (warningDescriptor) => {
        PRINT_WARNING(warningDescriptor.message);
      });
      this.TRACE_INIT("Choosing sub-methods implementations", () => {
        if (SUPPORT_STICKY) {
          this.chopInput = identity_default;
          this.match = this.matchWithTest;
        } else {
          this.updateLastIndex = noop_default;
          this.match = this.matchWithExec;
        }
        if (hasOnlySingleMode) {
          this.handleModes = noop_default;
        }
        if (this.trackStartLines === false) {
          this.computeNewColumn = identity_default;
        }
        if (this.trackEndLines === false) {
          this.updateTokenEndLineColumnLocation = noop_default;
        }
        if (/full/i.test(this.config.positionTracking)) {
          this.createTokenInstance = this.createFullToken;
        } else if (/onlyStart/i.test(this.config.positionTracking)) {
          this.createTokenInstance = this.createStartOnlyToken;
        } else if (/onlyOffset/i.test(this.config.positionTracking)) {
          this.createTokenInstance = this.createOffsetOnlyToken;
        } else {
          throw Error(`Invalid <positionTracking> config option: "${this.config.positionTracking}"`);
        }
        if (this.hasCustom) {
          this.addToken = this.addTokenUsingPush;
          this.handlePayload = this.handlePayloadWithCustom;
        } else {
          this.addToken = this.addTokenUsingMemberAccess;
          this.handlePayload = this.handlePayloadNoCustom;
        }
      });
      this.TRACE_INIT("Failed Optimization Warnings", () => {
        const unOptimizedModes = reduce_default(this.canModeBeOptimized, (cannotBeOptimized, canBeOptimized, modeName) => {
          if (canBeOptimized === false) {
            cannotBeOptimized.push(modeName);
          }
          return cannotBeOptimized;
        }, []);
        if (config.ensureOptimizations && !isEmpty_default(unOptimizedModes)) {
          throw Error(`Lexer Modes: < ${unOptimizedModes.join(", ")} > cannot be optimized.
	 Disable the "ensureOptimizations" lexer config flag to silently ignore this and run the lexer in an un-optimized mode.
	 Or inspect the console log for details on how to resolve these issues.`);
        }
      });
      this.TRACE_INIT("clearRegExpParserCache", () => {
        clearRegExpParserCache();
      });
      this.TRACE_INIT("toFastProperties", () => {
        toFastProperties(this);
      });
    });
  }
  tokenize(text, initialMode = this.defaultMode) {
    if (!isEmpty_default(this.lexerDefinitionErrors)) {
      const allErrMessages = map_default(this.lexerDefinitionErrors, (error) => {
        return error.message;
      });
      const allErrMessagesString = allErrMessages.join("-----------------------\n");
      throw new Error("Unable to Tokenize because Errors detected in definition of Lexer:\n" + allErrMessagesString);
    }
    return this.tokenizeInternal(text, initialMode);
  }
  // There is quite a bit of duplication between this and "tokenizeInternalLazy"
  // This is intentional due to performance considerations.
  // this method also used quite a bit of `!` none null assertions because it is too optimized
  // for `tsc` to always understand it is "safe"
  tokenizeInternal(text, initialMode) {
    let i, j, k, matchAltImage, longerAlt, matchedImage, payload, altPayload, imageLength, group, tokType, newToken, errLength, droppedChar, msg, match;
    const orgText = text;
    const orgLength = orgText.length;
    let offset = 0;
    let matchedTokensIndex = 0;
    const guessedNumberOfTokens = this.hasCustom ? 0 : Math.floor(text.length / 10);
    const matchedTokens = new Array(guessedNumberOfTokens);
    const errors = [];
    let line = this.trackStartLines ? 1 : void 0;
    let column = this.trackStartLines ? 1 : void 0;
    const groups = cloneEmptyGroups(this.emptyGroups);
    const trackLines = this.trackStartLines;
    const lineTerminatorPattern = this.config.lineTerminatorsPattern;
    let currModePatternsLength = 0;
    let patternIdxToConfig = [];
    let currCharCodeToPatternIdxToConfig = [];
    const modeStack = [];
    const emptyArray = [];
    Object.freeze(emptyArray);
    let getPossiblePatterns;
    function getPossiblePatternsSlow() {
      return patternIdxToConfig;
    }
    function getPossiblePatternsOptimized(charCode) {
      const optimizedCharIdx = charCodeToOptimizedIndex(charCode);
      const possiblePatterns = currCharCodeToPatternIdxToConfig[optimizedCharIdx];
      if (possiblePatterns === void 0) {
        return emptyArray;
      } else {
        return possiblePatterns;
      }
    }
    const pop_mode = (popToken) => {
      if (modeStack.length === 1 && // if we have both a POP_MODE and a PUSH_MODE this is in-fact a "transition"
      // So no error should occur.
      popToken.tokenType.PUSH_MODE === void 0) {
        const msg2 = this.config.errorMessageProvider.buildUnableToPopLexerModeMessage(popToken);
        errors.push({
          offset: popToken.startOffset,
          line: popToken.startLine,
          column: popToken.startColumn,
          length: popToken.image.length,
          message: msg2
        });
      } else {
        modeStack.pop();
        const newMode = last_default(modeStack);
        patternIdxToConfig = this.patternIdxToConfig[newMode];
        currCharCodeToPatternIdxToConfig = this.charCodeToPatternIdxToConfig[newMode];
        currModePatternsLength = patternIdxToConfig.length;
        const modeCanBeOptimized = this.canModeBeOptimized[newMode] && this.config.safeMode === false;
        if (currCharCodeToPatternIdxToConfig && modeCanBeOptimized) {
          getPossiblePatterns = getPossiblePatternsOptimized;
        } else {
          getPossiblePatterns = getPossiblePatternsSlow;
        }
      }
    };
    function push_mode(newMode) {
      modeStack.push(newMode);
      currCharCodeToPatternIdxToConfig = this.charCodeToPatternIdxToConfig[newMode];
      patternIdxToConfig = this.patternIdxToConfig[newMode];
      currModePatternsLength = patternIdxToConfig.length;
      currModePatternsLength = patternIdxToConfig.length;
      const modeCanBeOptimized = this.canModeBeOptimized[newMode] && this.config.safeMode === false;
      if (currCharCodeToPatternIdxToConfig && modeCanBeOptimized) {
        getPossiblePatterns = getPossiblePatternsOptimized;
      } else {
        getPossiblePatterns = getPossiblePatternsSlow;
      }
    }
    push_mode.call(this, initialMode);
    let currConfig;
    const recoveryEnabled = this.config.recoveryEnabled;
    while (offset < orgLength) {
      matchedImage = null;
      const nextCharCode = orgText.charCodeAt(offset);
      const chosenPatternIdxToConfig = getPossiblePatterns(nextCharCode);
      const chosenPatternsLength = chosenPatternIdxToConfig.length;
      for (i = 0; i < chosenPatternsLength; i++) {
        currConfig = chosenPatternIdxToConfig[i];
        const currPattern = currConfig.pattern;
        payload = null;
        const singleCharCode = currConfig.short;
        if (singleCharCode !== false) {
          if (nextCharCode === singleCharCode) {
            matchedImage = currPattern;
          }
        } else if (currConfig.isCustom === true) {
          match = currPattern.exec(orgText, offset, matchedTokens, groups);
          if (match !== null) {
            matchedImage = match[0];
            if (match.payload !== void 0) {
              payload = match.payload;
            }
          } else {
            matchedImage = null;
          }
        } else {
          this.updateLastIndex(currPattern, offset);
          matchedImage = this.match(currPattern, text, offset);
        }
        if (matchedImage !== null) {
          longerAlt = currConfig.longerAlt;
          if (longerAlt !== void 0) {
            const longerAltLength = longerAlt.length;
            for (k = 0; k < longerAltLength; k++) {
              const longerAltConfig = patternIdxToConfig[longerAlt[k]];
              const longerAltPattern = longerAltConfig.pattern;
              altPayload = null;
              if (longerAltConfig.isCustom === true) {
                match = longerAltPattern.exec(orgText, offset, matchedTokens, groups);
                if (match !== null) {
                  matchAltImage = match[0];
                  if (match.payload !== void 0) {
                    altPayload = match.payload;
                  }
                } else {
                  matchAltImage = null;
                }
              } else {
                this.updateLastIndex(longerAltPattern, offset);
                matchAltImage = this.match(longerAltPattern, text, offset);
              }
              if (matchAltImage && matchAltImage.length > matchedImage.length) {
                matchedImage = matchAltImage;
                payload = altPayload;
                currConfig = longerAltConfig;
                break;
              }
            }
          }
          break;
        }
      }
      if (matchedImage !== null) {
        imageLength = matchedImage.length;
        group = currConfig.group;
        if (group !== void 0) {
          tokType = currConfig.tokenTypeIdx;
          newToken = this.createTokenInstance(matchedImage, offset, tokType, currConfig.tokenType, line, column, imageLength);
          this.handlePayload(newToken, payload);
          if (group === false) {
            matchedTokensIndex = this.addToken(matchedTokens, matchedTokensIndex, newToken);
          } else {
            groups[group].push(newToken);
          }
        }
        text = this.chopInput(text, imageLength);
        offset = offset + imageLength;
        column = this.computeNewColumn(column, imageLength);
        if (trackLines === true && currConfig.canLineTerminator === true) {
          let numOfLTsInMatch = 0;
          let foundTerminator;
          let lastLTEndOffset;
          lineTerminatorPattern.lastIndex = 0;
          do {
            foundTerminator = lineTerminatorPattern.test(matchedImage);
            if (foundTerminator === true) {
              lastLTEndOffset = lineTerminatorPattern.lastIndex - 1;
              numOfLTsInMatch++;
            }
          } while (foundTerminator === true);
          if (numOfLTsInMatch !== 0) {
            line = line + numOfLTsInMatch;
            column = imageLength - lastLTEndOffset;
            this.updateTokenEndLineColumnLocation(newToken, group, lastLTEndOffset, numOfLTsInMatch, line, column, imageLength);
          }
        }
        this.handleModes(currConfig, pop_mode, push_mode, newToken);
      } else {
        const errorStartOffset = offset;
        const errorLine = line;
        const errorColumn = column;
        let foundResyncPoint = recoveryEnabled === false;
        while (foundResyncPoint === false && offset < orgLength) {
          text = this.chopInput(text, 1);
          offset++;
          for (j = 0; j < currModePatternsLength; j++) {
            const currConfig2 = patternIdxToConfig[j];
            const currPattern = currConfig2.pattern;
            const singleCharCode = currConfig2.short;
            if (singleCharCode !== false) {
              if (orgText.charCodeAt(offset) === singleCharCode) {
                foundResyncPoint = true;
              }
            } else if (currConfig2.isCustom === true) {
              foundResyncPoint = currPattern.exec(orgText, offset, matchedTokens, groups) !== null;
            } else {
              this.updateLastIndex(currPattern, offset);
              foundResyncPoint = currPattern.exec(text) !== null;
            }
            if (foundResyncPoint === true) {
              break;
            }
          }
        }
        errLength = offset - errorStartOffset;
        column = this.computeNewColumn(column, errLength);
        msg = this.config.errorMessageProvider.buildUnexpectedCharactersMessage(orgText, errorStartOffset, errLength, errorLine, errorColumn);
        errors.push({
          offset: errorStartOffset,
          line: errorLine,
          column: errorColumn,
          length: errLength,
          message: msg
        });
        if (recoveryEnabled === false) {
          break;
        }
      }
    }
    if (!this.hasCustom) {
      matchedTokens.length = matchedTokensIndex;
    }
    return {
      tokens: matchedTokens,
      groups,
      errors
    };
  }
  handleModes(config, pop_mode, push_mode, newToken) {
    if (config.pop === true) {
      const pushMode = config.push;
      pop_mode(newToken);
      if (pushMode !== void 0) {
        push_mode.call(this, pushMode);
      }
    } else if (config.push !== void 0) {
      push_mode.call(this, config.push);
    }
  }
  chopInput(text, length) {
    return text.substring(length);
  }
  updateLastIndex(regExp, newLastIndex) {
    regExp.lastIndex = newLastIndex;
  }
  // TODO: decrease this under 600 characters? inspect stripping comments option in TSC compiler
  updateTokenEndLineColumnLocation(newToken, group, lastLTIdx, numOfLTsInMatch, line, column, imageLength) {
    let lastCharIsLT, fixForEndingInLT;
    if (group !== void 0) {
      lastCharIsLT = lastLTIdx === imageLength - 1;
      fixForEndingInLT = lastCharIsLT ? -1 : 0;
      if (!(numOfLTsInMatch === 1 && lastCharIsLT === true)) {
        newToken.endLine = line + fixForEndingInLT;
        newToken.endColumn = column - 1 + -fixForEndingInLT;
      }
    }
  }
  computeNewColumn(oldColumn, imageLength) {
    return oldColumn + imageLength;
  }
  createOffsetOnlyToken(image, startOffset, tokenTypeIdx, tokenType) {
    return {
      image,
      startOffset,
      tokenTypeIdx,
      tokenType
    };
  }
  createStartOnlyToken(image, startOffset, tokenTypeIdx, tokenType, startLine, startColumn) {
    return {
      image,
      startOffset,
      startLine,
      startColumn,
      tokenTypeIdx,
      tokenType
    };
  }
  createFullToken(image, startOffset, tokenTypeIdx, tokenType, startLine, startColumn, imageLength) {
    return {
      image,
      startOffset,
      endOffset: startOffset + imageLength - 1,
      startLine,
      endLine: startLine,
      startColumn,
      endColumn: startColumn + imageLength - 1,
      tokenTypeIdx,
      tokenType
    };
  }
  addTokenUsingPush(tokenVector, index, tokenToAdd) {
    tokenVector.push(tokenToAdd);
    return index;
  }
  addTokenUsingMemberAccess(tokenVector, index, tokenToAdd) {
    tokenVector[index] = tokenToAdd;
    index++;
    return index;
  }
  handlePayloadNoCustom(token, payload) {
  }
  handlePayloadWithCustom(token, payload) {
    if (payload !== null) {
      token.payload = payload;
    }
  }
  matchWithTest(pattern, text, offset) {
    const found = pattern.test(text);
    if (found === true) {
      return text.substring(offset, pattern.lastIndex);
    }
    return null;
  }
  matchWithExec(pattern, text) {
    const regExpArray = pattern.exec(text);
    return regExpArray !== null ? regExpArray[0] : null;
  }
};
Lexer.SKIPPED = "This marks a skipped Token pattern, this means each token identified by it willbe consumed and then thrown into oblivion, this can be used to for example to completely ignore whitespace.";
Lexer.NA = /NOT_APPLICABLE/;

// node_modules/chevrotain/lib/src/scan/tokens_public.js
function tokenLabel2(tokType) {
  if (hasTokenLabel2(tokType)) {
    return tokType.LABEL;
  } else {
    return tokType.name;
  }
}
function hasTokenLabel2(obj) {
  return isString_default(obj.LABEL) && obj.LABEL !== "";
}
var PARENT = "parent";
var CATEGORIES = "categories";
var LABEL = "label";
var GROUP = "group";
var PUSH_MODE = "push_mode";
var POP_MODE = "pop_mode";
var LONGER_ALT = "longer_alt";
var LINE_BREAKS = "line_breaks";
var START_CHARS_HINT = "start_chars_hint";
function createToken(config) {
  return createTokenInternal(config);
}
function createTokenInternal(config) {
  const pattern = config.pattern;
  const tokenType = {};
  tokenType.name = config.name;
  if (!isUndefined_default(pattern)) {
    tokenType.PATTERN = pattern;
  }
  if (has_default(config, PARENT)) {
    throw "The parent property is no longer supported.\nSee: https://github.com/chevrotain/chevrotain/issues/564#issuecomment-349062346 for details.";
  }
  if (has_default(config, CATEGORIES)) {
    tokenType.CATEGORIES = config[CATEGORIES];
  }
  augmentTokenTypes([tokenType]);
  if (has_default(config, LABEL)) {
    tokenType.LABEL = config[LABEL];
  }
  if (has_default(config, GROUP)) {
    tokenType.GROUP = config[GROUP];
  }
  if (has_default(config, POP_MODE)) {
    tokenType.POP_MODE = config[POP_MODE];
  }
  if (has_default(config, PUSH_MODE)) {
    tokenType.PUSH_MODE = config[PUSH_MODE];
  }
  if (has_default(config, LONGER_ALT)) {
    tokenType.LONGER_ALT = config[LONGER_ALT];
  }
  if (has_default(config, LINE_BREAKS)) {
    tokenType.LINE_BREAKS = config[LINE_BREAKS];
  }
  if (has_default(config, START_CHARS_HINT)) {
    tokenType.START_CHARS_HINT = config[START_CHARS_HINT];
  }
  return tokenType;
}
var EOF = createToken({ name: "EOF", pattern: Lexer.NA });
augmentTokenTypes([EOF]);
function createTokenInstance(tokType, image, startOffset, endOffset, startLine, endLine, startColumn, endColumn) {
  return {
    image,
    startOffset,
    endOffset,
    startLine,
    endLine,
    startColumn,
    endColumn,
    tokenTypeIdx: tokType.tokenTypeIdx,
    tokenType: tokType
  };
}
function tokenMatcher(token, tokType) {
  return tokenStructuredMatcher(token, tokType);
}

// node_modules/chevrotain/lib/src/parse/errors_public.js
var defaultParserErrorProvider = {
  buildMismatchTokenMessage({ expected, actual, previous, ruleName }) {
    const hasLabel = hasTokenLabel2(expected);
    const expectedMsg = hasLabel ? `--> ${tokenLabel2(expected)} <--` : `token of type --> ${expected.name} <--`;
    const msg = `Expecting ${expectedMsg} but found --> '${actual.image}' <--`;
    return msg;
  },
  buildNotAllInputParsedMessage({ firstRedundant, ruleName }) {
    return "Redundant input, expecting EOF but found: " + firstRedundant.image;
  },
  buildNoViableAltMessage({ expectedPathsPerAlt, actual, previous, customUserDescription, ruleName }) {
    const errPrefix = "Expecting: ";
    const actualText = head_default(actual).image;
    const errSuffix = "\nbut found: '" + actualText + "'";
    if (customUserDescription) {
      return errPrefix + customUserDescription + errSuffix;
    } else {
      const allLookAheadPaths = reduce_default(expectedPathsPerAlt, (result2, currAltPaths) => result2.concat(currAltPaths), []);
      const nextValidTokenSequences = map_default(allLookAheadPaths, (currPath) => `[${map_default(currPath, (currTokenType) => tokenLabel2(currTokenType)).join(", ")}]`);
      const nextValidSequenceItems = map_default(nextValidTokenSequences, (itemMsg, idx) => `  ${idx + 1}. ${itemMsg}`);
      const calculatedDescription = `one of these possible Token sequences:
${nextValidSequenceItems.join("\n")}`;
      return errPrefix + calculatedDescription + errSuffix;
    }
  },
  buildEarlyExitMessage({ expectedIterationPaths, actual, customUserDescription, ruleName }) {
    const errPrefix = "Expecting: ";
    const actualText = head_default(actual).image;
    const errSuffix = "\nbut found: '" + actualText + "'";
    if (customUserDescription) {
      return errPrefix + customUserDescription + errSuffix;
    } else {
      const nextValidTokenSequences = map_default(expectedIterationPaths, (currPath) => `[${map_default(currPath, (currTokenType) => tokenLabel2(currTokenType)).join(",")}]`);
      const calculatedDescription = `expecting at least one iteration which starts with one of these possible Token sequences::
  <${nextValidTokenSequences.join(" ,")}>`;
      return errPrefix + calculatedDescription + errSuffix;
    }
  }
};
Object.freeze(defaultParserErrorProvider);
var defaultGrammarResolverErrorProvider = {
  buildRuleNotFoundError(topLevelRule, undefinedRule) {
    const msg = "Invalid grammar, reference to a rule which is not defined: ->" + undefinedRule.nonTerminalName + "<-\ninside top level rule: ->" + topLevelRule.name + "<-";
    return msg;
  }
};
var defaultGrammarValidatorErrorProvider = {
  buildDuplicateFoundError(topLevelRule, duplicateProds) {
    function getExtraProductionArgument2(prod) {
      if (prod instanceof Terminal) {
        return prod.terminalType.name;
      } else if (prod instanceof NonTerminal) {
        return prod.nonTerminalName;
      } else {
        return "";
      }
    }
    const topLevelName = topLevelRule.name;
    const duplicateProd = head_default(duplicateProds);
    const index = duplicateProd.idx;
    const dslName = getProductionDslName(duplicateProd);
    const extraArgument = getExtraProductionArgument2(duplicateProd);
    const hasExplicitIndex = index > 0;
    let msg = `->${dslName}${hasExplicitIndex ? index : ""}<- ${extraArgument ? `with argument: ->${extraArgument}<-` : ""}
                  appears more than once (${duplicateProds.length} times) in the top level rule: ->${topLevelName}<-.                  
                  For further details see: https://chevrotain.io/docs/FAQ.html#NUMERICAL_SUFFIXES 
                  `;
    msg = msg.replace(/[ \t]+/g, " ");
    msg = msg.replace(/\s\s+/g, "\n");
    return msg;
  },
  buildNamespaceConflictError(rule) {
    const errMsg = `Namespace conflict found in grammar.
The grammar has both a Terminal(Token) and a Non-Terminal(Rule) named: <${rule.name}>.
To resolve this make sure each Terminal and Non-Terminal names are unique
This is easy to accomplish by using the convention that Terminal names start with an uppercase letter
and Non-Terminal names start with a lower case letter.`;
    return errMsg;
  },
  buildAlternationPrefixAmbiguityError(options) {
    const pathMsg = map_default(options.prefixPath, (currTok) => tokenLabel2(currTok)).join(", ");
    const occurrence = options.alternation.idx === 0 ? "" : options.alternation.idx;
    const errMsg = `Ambiguous alternatives: <${options.ambiguityIndices.join(" ,")}> due to common lookahead prefix
in <OR${occurrence}> inside <${options.topLevelRule.name}> Rule,
<${pathMsg}> may appears as a prefix path in all these alternatives.
See: https://chevrotain.io/docs/guide/resolving_grammar_errors.html#COMMON_PREFIX
For Further details.`;
    return errMsg;
  },
  buildAlternationAmbiguityError(options) {
    const pathMsg = map_default(options.prefixPath, (currtok) => tokenLabel2(currtok)).join(", ");
    const occurrence = options.alternation.idx === 0 ? "" : options.alternation.idx;
    let currMessage = `Ambiguous Alternatives Detected: <${options.ambiguityIndices.join(" ,")}> in <OR${occurrence}> inside <${options.topLevelRule.name}> Rule,
<${pathMsg}> may appears as a prefix path in all these alternatives.
`;
    currMessage = currMessage + `See: https://chevrotain.io/docs/guide/resolving_grammar_errors.html#AMBIGUOUS_ALTERNATIVES
For Further details.`;
    return currMessage;
  },
  buildEmptyRepetitionError(options) {
    let dslName = getProductionDslName(options.repetition);
    if (options.repetition.idx !== 0) {
      dslName += options.repetition.idx;
    }
    const errMsg = `The repetition <${dslName}> within Rule <${options.topLevelRule.name}> can never consume any tokens.
This could lead to an infinite loop.`;
    return errMsg;
  },
  // TODO: remove - `errors_public` from nyc.config.js exclude
  //       once this method is fully removed from this file
  buildTokenNameError(options) {
    return "deprecated";
  },
  buildEmptyAlternationError(options) {
    const errMsg = `Ambiguous empty alternative: <${options.emptyChoiceIdx + 1}> in <OR${options.alternation.idx}> inside <${options.topLevelRule.name}> Rule.
Only the last alternative may be an empty alternative.`;
    return errMsg;
  },
  buildTooManyAlternativesError(options) {
    const errMsg = `An Alternation cannot have more than 256 alternatives:
<OR${options.alternation.idx}> inside <${options.topLevelRule.name}> Rule.
 has ${options.alternation.definition.length + 1} alternatives.`;
    return errMsg;
  },
  buildLeftRecursionError(options) {
    const ruleName = options.topLevelRule.name;
    const pathNames = map_default(options.leftRecursionPath, (currRule) => currRule.name);
    const leftRecursivePath = `${ruleName} --> ${pathNames.concat([ruleName]).join(" --> ")}`;
    const errMsg = `Left Recursion found in grammar.
rule: <${ruleName}> can be invoked from itself (directly or indirectly)
without consuming any Tokens. The grammar path that causes this is: 
 ${leftRecursivePath}
 To fix this refactor your grammar to remove the left recursion.
see: https://en.wikipedia.org/wiki/LL_parser#Left_factoring.`;
    return errMsg;
  },
  // TODO: remove - `errors_public` from nyc.config.js exclude
  //       once this method is fully removed from this file
  buildInvalidRuleNameError(options) {
    return "deprecated";
  },
  buildDuplicateRuleNameError(options) {
    let ruleName;
    if (options.topLevelRule instanceof Rule) {
      ruleName = options.topLevelRule.name;
    } else {
      ruleName = options.topLevelRule;
    }
    const errMsg = `Duplicate definition, rule: ->${ruleName}<- is already defined in the grammar: ->${options.grammarName}<-`;
    return errMsg;
  }
};

// node_modules/chevrotain/lib/src/parse/grammar/resolver.js
function resolveGrammar(topLevels, errMsgProvider) {
  const refResolver = new GastRefResolverVisitor(topLevels, errMsgProvider);
  refResolver.resolveRefs();
  return refResolver.errors;
}
var GastRefResolverVisitor = class extends GAstVisitor {
  constructor(nameToTopRule, errMsgProvider) {
    super();
    this.nameToTopRule = nameToTopRule;
    this.errMsgProvider = errMsgProvider;
    this.errors = [];
  }
  resolveRefs() {
    forEach_default(values_default(this.nameToTopRule), (prod) => {
      this.currTopLevel = prod;
      prod.accept(this);
    });
  }
  visitNonTerminal(node) {
    const ref = this.nameToTopRule[node.nonTerminalName];
    if (!ref) {
      const msg = this.errMsgProvider.buildRuleNotFoundError(this.currTopLevel, node);
      this.errors.push({
        message: msg,
        type: ParserDefinitionErrorType.UNRESOLVED_SUBRULE_REF,
        ruleName: this.currTopLevel.name,
        unresolvedRefName: node.nonTerminalName
      });
    } else {
      node.referencedRule = ref;
    }
  }
};

// node_modules/chevrotain/lib/src/parse/grammar/interpreter.js
var AbstractNextPossibleTokensWalker = class extends RestWalker {
  constructor(topProd, path) {
    super();
    this.topProd = topProd;
    this.path = path;
    this.possibleTokTypes = [];
    this.nextProductionName = "";
    this.nextProductionOccurrence = 0;
    this.found = false;
    this.isAtEndOfPath = false;
  }
  startWalking() {
    this.found = false;
    if (this.path.ruleStack[0] !== this.topProd.name) {
      throw Error("The path does not start with the walker's top Rule!");
    }
    this.ruleStack = clone_default(this.path.ruleStack).reverse();
    this.occurrenceStack = clone_default(this.path.occurrenceStack).reverse();
    this.ruleStack.pop();
    this.occurrenceStack.pop();
    this.updateExpectedNext();
    this.walk(this.topProd);
    return this.possibleTokTypes;
  }
  walk(prod, prevRest = []) {
    if (!this.found) {
      super.walk(prod, prevRest);
    }
  }
  walkProdRef(refProd, currRest, prevRest) {
    if (refProd.referencedRule.name === this.nextProductionName && refProd.idx === this.nextProductionOccurrence) {
      const fullRest = currRest.concat(prevRest);
      this.updateExpectedNext();
      this.walk(refProd.referencedRule, fullRest);
    }
  }
  updateExpectedNext() {
    if (isEmpty_default(this.ruleStack)) {
      this.nextProductionName = "";
      this.nextProductionOccurrence = 0;
      this.isAtEndOfPath = true;
    } else {
      this.nextProductionName = this.ruleStack.pop();
      this.nextProductionOccurrence = this.occurrenceStack.pop();
    }
  }
};
var NextAfterTokenWalker = class extends AbstractNextPossibleTokensWalker {
  constructor(topProd, path) {
    super(topProd, path);
    this.path = path;
    this.nextTerminalName = "";
    this.nextTerminalOccurrence = 0;
    this.nextTerminalName = this.path.lastTok.name;
    this.nextTerminalOccurrence = this.path.lastTokOccurrence;
  }
  walkTerminal(terminal, currRest, prevRest) {
    if (this.isAtEndOfPath && terminal.terminalType.name === this.nextTerminalName && terminal.idx === this.nextTerminalOccurrence && !this.found) {
      const fullRest = currRest.concat(prevRest);
      const restProd = new Alternative({ definition: fullRest });
      this.possibleTokTypes = first(restProd);
      this.found = true;
    }
  }
};
var AbstractNextTerminalAfterProductionWalker = class extends RestWalker {
  constructor(topRule, occurrence) {
    super();
    this.topRule = topRule;
    this.occurrence = occurrence;
    this.result = {
      token: void 0,
      occurrence: void 0,
      isEndOfRule: void 0
    };
  }
  startWalking() {
    this.walk(this.topRule);
    return this.result;
  }
};
var NextTerminalAfterManyWalker = class extends AbstractNextTerminalAfterProductionWalker {
  walkMany(manyProd, currRest, prevRest) {
    if (manyProd.idx === this.occurrence) {
      const firstAfterMany = head_default(currRest.concat(prevRest));
      this.result.isEndOfRule = firstAfterMany === void 0;
      if (firstAfterMany instanceof Terminal) {
        this.result.token = firstAfterMany.terminalType;
        this.result.occurrence = firstAfterMany.idx;
      }
    } else {
      super.walkMany(manyProd, currRest, prevRest);
    }
  }
};
var NextTerminalAfterManySepWalker = class extends AbstractNextTerminalAfterProductionWalker {
  walkManySep(manySepProd, currRest, prevRest) {
    if (manySepProd.idx === this.occurrence) {
      const firstAfterManySep = head_default(currRest.concat(prevRest));
      this.result.isEndOfRule = firstAfterManySep === void 0;
      if (firstAfterManySep instanceof Terminal) {
        this.result.token = firstAfterManySep.terminalType;
        this.result.occurrence = firstAfterManySep.idx;
      }
    } else {
      super.walkManySep(manySepProd, currRest, prevRest);
    }
  }
};
var NextTerminalAfterAtLeastOneWalker = class extends AbstractNextTerminalAfterProductionWalker {
  walkAtLeastOne(atLeastOneProd, currRest, prevRest) {
    if (atLeastOneProd.idx === this.occurrence) {
      const firstAfterAtLeastOne = head_default(currRest.concat(prevRest));
      this.result.isEndOfRule = firstAfterAtLeastOne === void 0;
      if (firstAfterAtLeastOne instanceof Terminal) {
        this.result.token = firstAfterAtLeastOne.terminalType;
        this.result.occurrence = firstAfterAtLeastOne.idx;
      }
    } else {
      super.walkAtLeastOne(atLeastOneProd, currRest, prevRest);
    }
  }
};
var NextTerminalAfterAtLeastOneSepWalker = class extends AbstractNextTerminalAfterProductionWalker {
  walkAtLeastOneSep(atleastOneSepProd, currRest, prevRest) {
    if (atleastOneSepProd.idx === this.occurrence) {
      const firstAfterfirstAfterAtLeastOneSep = head_default(currRest.concat(prevRest));
      this.result.isEndOfRule = firstAfterfirstAfterAtLeastOneSep === void 0;
      if (firstAfterfirstAfterAtLeastOneSep instanceof Terminal) {
        this.result.token = firstAfterfirstAfterAtLeastOneSep.terminalType;
        this.result.occurrence = firstAfterfirstAfterAtLeastOneSep.idx;
      }
    } else {
      super.walkAtLeastOneSep(atleastOneSepProd, currRest, prevRest);
    }
  }
};
function possiblePathsFrom(targetDef, maxLength, currPath = []) {
  currPath = clone_default(currPath);
  let result2 = [];
  let i = 0;
  function remainingPathWith(nextDef) {
    return nextDef.concat(drop_default(targetDef, i + 1));
  }
  function getAlternativesForProd(definition) {
    const alternatives = possiblePathsFrom(remainingPathWith(definition), maxLength, currPath);
    return result2.concat(alternatives);
  }
  while (currPath.length < maxLength && i < targetDef.length) {
    const prod = targetDef[i];
    if (prod instanceof Alternative) {
      return getAlternativesForProd(prod.definition);
    } else if (prod instanceof NonTerminal) {
      return getAlternativesForProd(prod.definition);
    } else if (prod instanceof Option) {
      result2 = getAlternativesForProd(prod.definition);
    } else if (prod instanceof RepetitionMandatory) {
      const newDef = prod.definition.concat([
        new Repetition({
          definition: prod.definition
        })
      ]);
      return getAlternativesForProd(newDef);
    } else if (prod instanceof RepetitionMandatoryWithSeparator) {
      const newDef = [
        new Alternative({ definition: prod.definition }),
        new Repetition({
          definition: [new Terminal({ terminalType: prod.separator })].concat(prod.definition)
        })
      ];
      return getAlternativesForProd(newDef);
    } else if (prod instanceof RepetitionWithSeparator) {
      const newDef = prod.definition.concat([
        new Repetition({
          definition: [new Terminal({ terminalType: prod.separator })].concat(prod.definition)
        })
      ]);
      result2 = getAlternativesForProd(newDef);
    } else if (prod instanceof Repetition) {
      const newDef = prod.definition.concat([
        new Repetition({
          definition: prod.definition
        })
      ]);
      result2 = getAlternativesForProd(newDef);
    } else if (prod instanceof Alternation) {
      forEach_default(prod.definition, (currAlt) => {
        if (isEmpty_default(currAlt.definition) === false) {
          result2 = getAlternativesForProd(currAlt.definition);
        }
      });
      return result2;
    } else if (prod instanceof Terminal) {
      currPath.push(prod.terminalType);
    } else {
      throw Error("non exhaustive match");
    }
    i++;
  }
  result2.push({
    partialPath: currPath,
    suffixDef: drop_default(targetDef, i)
  });
  return result2;
}
function nextPossibleTokensAfter(initialDef, tokenVector, tokMatcher, maxLookAhead) {
  const EXIT_NON_TERMINAL = "EXIT_NONE_TERMINAL";
  const EXIT_NON_TERMINAL_ARR = [EXIT_NON_TERMINAL];
  const EXIT_ALTERNATIVE = "EXIT_ALTERNATIVE";
  let foundCompletePath = false;
  const tokenVectorLength = tokenVector.length;
  const minimalAlternativesIndex = tokenVectorLength - maxLookAhead - 1;
  const result2 = [];
  const possiblePaths = [];
  possiblePaths.push({
    idx: -1,
    def: initialDef,
    ruleStack: [],
    occurrenceStack: []
  });
  while (!isEmpty_default(possiblePaths)) {
    const currPath = possiblePaths.pop();
    if (currPath === EXIT_ALTERNATIVE) {
      if (foundCompletePath && last_default(possiblePaths).idx <= minimalAlternativesIndex) {
        possiblePaths.pop();
      }
      continue;
    }
    const currDef = currPath.def;
    const currIdx = currPath.idx;
    const currRuleStack = currPath.ruleStack;
    const currOccurrenceStack = currPath.occurrenceStack;
    if (isEmpty_default(currDef)) {
      continue;
    }
    const prod = currDef[0];
    if (prod === EXIT_NON_TERMINAL) {
      const nextPath = {
        idx: currIdx,
        def: drop_default(currDef),
        ruleStack: dropRight_default(currRuleStack),
        occurrenceStack: dropRight_default(currOccurrenceStack)
      };
      possiblePaths.push(nextPath);
    } else if (prod instanceof Terminal) {
      if (currIdx < tokenVectorLength - 1) {
        const nextIdx = currIdx + 1;
        const actualToken = tokenVector[nextIdx];
        if (tokMatcher(actualToken, prod.terminalType)) {
          const nextPath = {
            idx: nextIdx,
            def: drop_default(currDef),
            ruleStack: currRuleStack,
            occurrenceStack: currOccurrenceStack
          };
          possiblePaths.push(nextPath);
        }
      } else if (currIdx === tokenVectorLength - 1) {
        result2.push({
          nextTokenType: prod.terminalType,
          nextTokenOccurrence: prod.idx,
          ruleStack: currRuleStack,
          occurrenceStack: currOccurrenceStack
        });
        foundCompletePath = true;
      } else {
        throw Error("non exhaustive match");
      }
    } else if (prod instanceof NonTerminal) {
      const newRuleStack = clone_default(currRuleStack);
      newRuleStack.push(prod.nonTerminalName);
      const newOccurrenceStack = clone_default(currOccurrenceStack);
      newOccurrenceStack.push(prod.idx);
      const nextPath = {
        idx: currIdx,
        def: prod.definition.concat(EXIT_NON_TERMINAL_ARR, drop_default(currDef)),
        ruleStack: newRuleStack,
        occurrenceStack: newOccurrenceStack
      };
      possiblePaths.push(nextPath);
    } else if (prod instanceof Option) {
      const nextPathWithout = {
        idx: currIdx,
        def: drop_default(currDef),
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPathWithout);
      possiblePaths.push(EXIT_ALTERNATIVE);
      const nextPathWith = {
        idx: currIdx,
        def: prod.definition.concat(drop_default(currDef)),
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPathWith);
    } else if (prod instanceof RepetitionMandatory) {
      const secondIteration = new Repetition({
        definition: prod.definition,
        idx: prod.idx
      });
      const nextDef = prod.definition.concat([secondIteration], drop_default(currDef));
      const nextPath = {
        idx: currIdx,
        def: nextDef,
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPath);
    } else if (prod instanceof RepetitionMandatoryWithSeparator) {
      const separatorGast = new Terminal({
        terminalType: prod.separator
      });
      const secondIteration = new Repetition({
        definition: [separatorGast].concat(prod.definition),
        idx: prod.idx
      });
      const nextDef = prod.definition.concat([secondIteration], drop_default(currDef));
      const nextPath = {
        idx: currIdx,
        def: nextDef,
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPath);
    } else if (prod instanceof RepetitionWithSeparator) {
      const nextPathWithout = {
        idx: currIdx,
        def: drop_default(currDef),
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPathWithout);
      possiblePaths.push(EXIT_ALTERNATIVE);
      const separatorGast = new Terminal({
        terminalType: prod.separator
      });
      const nthRepetition = new Repetition({
        definition: [separatorGast].concat(prod.definition),
        idx: prod.idx
      });
      const nextDef = prod.definition.concat([nthRepetition], drop_default(currDef));
      const nextPathWith = {
        idx: currIdx,
        def: nextDef,
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPathWith);
    } else if (prod instanceof Repetition) {
      const nextPathWithout = {
        idx: currIdx,
        def: drop_default(currDef),
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPathWithout);
      possiblePaths.push(EXIT_ALTERNATIVE);
      const nthRepetition = new Repetition({
        definition: prod.definition,
        idx: prod.idx
      });
      const nextDef = prod.definition.concat([nthRepetition], drop_default(currDef));
      const nextPathWith = {
        idx: currIdx,
        def: nextDef,
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      };
      possiblePaths.push(nextPathWith);
    } else if (prod instanceof Alternation) {
      for (let i = prod.definition.length - 1; i >= 0; i--) {
        const currAlt = prod.definition[i];
        const currAltPath = {
          idx: currIdx,
          def: currAlt.definition.concat(drop_default(currDef)),
          ruleStack: currRuleStack,
          occurrenceStack: currOccurrenceStack
        };
        possiblePaths.push(currAltPath);
        possiblePaths.push(EXIT_ALTERNATIVE);
      }
    } else if (prod instanceof Alternative) {
      possiblePaths.push({
        idx: currIdx,
        def: prod.definition.concat(drop_default(currDef)),
        ruleStack: currRuleStack,
        occurrenceStack: currOccurrenceStack
      });
    } else if (prod instanceof Rule) {
      possiblePaths.push(expandTopLevelRule(prod, currIdx, currRuleStack, currOccurrenceStack));
    } else {
      throw Error("non exhaustive match");
    }
  }
  return result2;
}
function expandTopLevelRule(topRule, currIdx, currRuleStack, currOccurrenceStack) {
  const newRuleStack = clone_default(currRuleStack);
  newRuleStack.push(topRule.name);
  const newCurrOccurrenceStack = clone_default(currOccurrenceStack);
  newCurrOccurrenceStack.push(1);
  return {
    idx: currIdx,
    def: topRule.definition,
    ruleStack: newRuleStack,
    occurrenceStack: newCurrOccurrenceStack
  };
}

// node_modules/chevrotain/lib/src/parse/grammar/lookahead.js
var PROD_TYPE;
(function(PROD_TYPE2) {
  PROD_TYPE2[PROD_TYPE2["OPTION"] = 0] = "OPTION";
  PROD_TYPE2[PROD_TYPE2["REPETITION"] = 1] = "REPETITION";
  PROD_TYPE2[PROD_TYPE2["REPETITION_MANDATORY"] = 2] = "REPETITION_MANDATORY";
  PROD_TYPE2[PROD_TYPE2["REPETITION_MANDATORY_WITH_SEPARATOR"] = 3] = "REPETITION_MANDATORY_WITH_SEPARATOR";
  PROD_TYPE2[PROD_TYPE2["REPETITION_WITH_SEPARATOR"] = 4] = "REPETITION_WITH_SEPARATOR";
  PROD_TYPE2[PROD_TYPE2["ALTERNATION"] = 5] = "ALTERNATION";
})(PROD_TYPE || (PROD_TYPE = {}));
function getProdType(prod) {
  if (prod instanceof Option || prod === "Option") {
    return PROD_TYPE.OPTION;
  } else if (prod instanceof Repetition || prod === "Repetition") {
    return PROD_TYPE.REPETITION;
  } else if (prod instanceof RepetitionMandatory || prod === "RepetitionMandatory") {
    return PROD_TYPE.REPETITION_MANDATORY;
  } else if (prod instanceof RepetitionMandatoryWithSeparator || prod === "RepetitionMandatoryWithSeparator") {
    return PROD_TYPE.REPETITION_MANDATORY_WITH_SEPARATOR;
  } else if (prod instanceof RepetitionWithSeparator || prod === "RepetitionWithSeparator") {
    return PROD_TYPE.REPETITION_WITH_SEPARATOR;
  } else if (prod instanceof Alternation || prod === "Alternation") {
    return PROD_TYPE.ALTERNATION;
  } else {
    throw Error("non exhaustive match");
  }
}
function getLookaheadPaths(options) {
  const { occurrence, rule, prodType, maxLookahead } = options;
  const type = getProdType(prodType);
  if (type === PROD_TYPE.ALTERNATION) {
    return getLookaheadPathsForOr(occurrence, rule, maxLookahead);
  } else {
    return getLookaheadPathsForOptionalProd(occurrence, rule, type, maxLookahead);
  }
}
function buildLookaheadFuncForOr(occurrence, ruleGrammar, maxLookahead, hasPredicates, dynamicTokensEnabled, laFuncBuilder) {
  const lookAheadPaths = getLookaheadPathsForOr(occurrence, ruleGrammar, maxLookahead);
  const tokenMatcher2 = areTokenCategoriesNotUsed(lookAheadPaths) ? tokenStructuredMatcherNoCategories : tokenStructuredMatcher;
  return laFuncBuilder(lookAheadPaths, hasPredicates, tokenMatcher2, dynamicTokensEnabled);
}
function buildLookaheadFuncForOptionalProd(occurrence, ruleGrammar, k, dynamicTokensEnabled, prodType, lookaheadBuilder) {
  const lookAheadPaths = getLookaheadPathsForOptionalProd(occurrence, ruleGrammar, prodType, k);
  const tokenMatcher2 = areTokenCategoriesNotUsed(lookAheadPaths) ? tokenStructuredMatcherNoCategories : tokenStructuredMatcher;
  return lookaheadBuilder(lookAheadPaths[0], tokenMatcher2, dynamicTokensEnabled);
}
function buildAlternativesLookAheadFunc(alts, hasPredicates, tokenMatcher2, dynamicTokensEnabled) {
  const numOfAlts = alts.length;
  const areAllOneTokenLookahead = every_default(alts, (currAlt) => {
    return every_default(currAlt, (currPath) => {
      return currPath.length === 1;
    });
  });
  if (hasPredicates) {
    return function(orAlts) {
      const predicates = map_default(orAlts, (currAlt) => currAlt.GATE);
      for (let t = 0; t < numOfAlts; t++) {
        const currAlt = alts[t];
        const currNumOfPaths = currAlt.length;
        const currPredicate = predicates[t];
        if (currPredicate !== void 0 && currPredicate.call(this) === false) {
          continue;
        }
        nextPath: for (let j = 0; j < currNumOfPaths; j++) {
          const currPath = currAlt[j];
          const currPathLength = currPath.length;
          for (let i = 0; i < currPathLength; i++) {
            const nextToken = this.LA(i + 1);
            if (tokenMatcher2(nextToken, currPath[i]) === false) {
              continue nextPath;
            }
          }
          return t;
        }
      }
      return void 0;
    };
  } else if (areAllOneTokenLookahead && !dynamicTokensEnabled) {
    const singleTokenAlts = map_default(alts, (currAlt) => {
      return flatten_default(currAlt);
    });
    const choiceToAlt = reduce_default(singleTokenAlts, (result2, currAlt, idx) => {
      forEach_default(currAlt, (currTokType) => {
        if (!has_default(result2, currTokType.tokenTypeIdx)) {
          result2[currTokType.tokenTypeIdx] = idx;
        }
        forEach_default(currTokType.categoryMatches, (currExtendingType) => {
          if (!has_default(result2, currExtendingType)) {
            result2[currExtendingType] = idx;
          }
        });
      });
      return result2;
    }, {});
    return function() {
      const nextToken = this.LA(1);
      return choiceToAlt[nextToken.tokenTypeIdx];
    };
  } else {
    return function() {
      for (let t = 0; t < numOfAlts; t++) {
        const currAlt = alts[t];
        const currNumOfPaths = currAlt.length;
        nextPath: for (let j = 0; j < currNumOfPaths; j++) {
          const currPath = currAlt[j];
          const currPathLength = currPath.length;
          for (let i = 0; i < currPathLength; i++) {
            const nextToken = this.LA(i + 1);
            if (tokenMatcher2(nextToken, currPath[i]) === false) {
              continue nextPath;
            }
          }
          return t;
        }
      }
      return void 0;
    };
  }
}
function buildSingleAlternativeLookaheadFunction(alt, tokenMatcher2, dynamicTokensEnabled) {
  const areAllOneTokenLookahead = every_default(alt, (currPath) => {
    return currPath.length === 1;
  });
  const numOfPaths = alt.length;
  if (areAllOneTokenLookahead && !dynamicTokensEnabled) {
    const singleTokensTypes = flatten_default(alt);
    if (singleTokensTypes.length === 1 && isEmpty_default(singleTokensTypes[0].categoryMatches)) {
      const expectedTokenType = singleTokensTypes[0];
      const expectedTokenUniqueKey = expectedTokenType.tokenTypeIdx;
      return function() {
        return this.LA(1).tokenTypeIdx === expectedTokenUniqueKey;
      };
    } else {
      const choiceToAlt = reduce_default(singleTokensTypes, (result2, currTokType, idx) => {
        result2[currTokType.tokenTypeIdx] = true;
        forEach_default(currTokType.categoryMatches, (currExtendingType) => {
          result2[currExtendingType] = true;
        });
        return result2;
      }, []);
      return function() {
        const nextToken = this.LA(1);
        return choiceToAlt[nextToken.tokenTypeIdx] === true;
      };
    }
  } else {
    return function() {
      nextPath: for (let j = 0; j < numOfPaths; j++) {
        const currPath = alt[j];
        const currPathLength = currPath.length;
        for (let i = 0; i < currPathLength; i++) {
          const nextToken = this.LA(i + 1);
          if (tokenMatcher2(nextToken, currPath[i]) === false) {
            continue nextPath;
          }
        }
        return true;
      }
      return false;
    };
  }
}
var RestDefinitionFinderWalker = class extends RestWalker {
  constructor(topProd, targetOccurrence, targetProdType) {
    super();
    this.topProd = topProd;
    this.targetOccurrence = targetOccurrence;
    this.targetProdType = targetProdType;
  }
  startWalking() {
    this.walk(this.topProd);
    return this.restDef;
  }
  checkIsTarget(node, expectedProdType, currRest, prevRest) {
    if (node.idx === this.targetOccurrence && this.targetProdType === expectedProdType) {
      this.restDef = currRest.concat(prevRest);
      return true;
    }
    return false;
  }
  walkOption(optionProd, currRest, prevRest) {
    if (!this.checkIsTarget(optionProd, PROD_TYPE.OPTION, currRest, prevRest)) {
      super.walkOption(optionProd, currRest, prevRest);
    }
  }
  walkAtLeastOne(atLeastOneProd, currRest, prevRest) {
    if (!this.checkIsTarget(atLeastOneProd, PROD_TYPE.REPETITION_MANDATORY, currRest, prevRest)) {
      super.walkOption(atLeastOneProd, currRest, prevRest);
    }
  }
  walkAtLeastOneSep(atLeastOneSepProd, currRest, prevRest) {
    if (!this.checkIsTarget(atLeastOneSepProd, PROD_TYPE.REPETITION_MANDATORY_WITH_SEPARATOR, currRest, prevRest)) {
      super.walkOption(atLeastOneSepProd, currRest, prevRest);
    }
  }
  walkMany(manyProd, currRest, prevRest) {
    if (!this.checkIsTarget(manyProd, PROD_TYPE.REPETITION, currRest, prevRest)) {
      super.walkOption(manyProd, currRest, prevRest);
    }
  }
  walkManySep(manySepProd, currRest, prevRest) {
    if (!this.checkIsTarget(manySepProd, PROD_TYPE.REPETITION_WITH_SEPARATOR, currRest, prevRest)) {
      super.walkOption(manySepProd, currRest, prevRest);
    }
  }
};
var InsideDefinitionFinderVisitor = class extends GAstVisitor {
  constructor(targetOccurrence, targetProdType, targetRef) {
    super();
    this.targetOccurrence = targetOccurrence;
    this.targetProdType = targetProdType;
    this.targetRef = targetRef;
    this.result = [];
  }
  checkIsTarget(node, expectedProdName) {
    if (node.idx === this.targetOccurrence && this.targetProdType === expectedProdName && (this.targetRef === void 0 || node === this.targetRef)) {
      this.result = node.definition;
    }
  }
  visitOption(node) {
    this.checkIsTarget(node, PROD_TYPE.OPTION);
  }
  visitRepetition(node) {
    this.checkIsTarget(node, PROD_TYPE.REPETITION);
  }
  visitRepetitionMandatory(node) {
    this.checkIsTarget(node, PROD_TYPE.REPETITION_MANDATORY);
  }
  visitRepetitionMandatoryWithSeparator(node) {
    this.checkIsTarget(node, PROD_TYPE.REPETITION_MANDATORY_WITH_SEPARATOR);
  }
  visitRepetitionWithSeparator(node) {
    this.checkIsTarget(node, PROD_TYPE.REPETITION_WITH_SEPARATOR);
  }
  visitAlternation(node) {
    this.checkIsTarget(node, PROD_TYPE.ALTERNATION);
  }
};
function initializeArrayOfArrays(size2) {
  const result2 = new Array(size2);
  for (let i = 0; i < size2; i++) {
    result2[i] = [];
  }
  return result2;
}
function pathToHashKeys(path) {
  let keys2 = [""];
  for (let i = 0; i < path.length; i++) {
    const tokType = path[i];
    const longerKeys = [];
    for (let j = 0; j < keys2.length; j++) {
      const currShorterKey = keys2[j];
      longerKeys.push(currShorterKey + "_" + tokType.tokenTypeIdx);
      for (let t = 0; t < tokType.categoryMatches.length; t++) {
        const categoriesKeySuffix = "_" + tokType.categoryMatches[t];
        longerKeys.push(currShorterKey + categoriesKeySuffix);
      }
    }
    keys2 = longerKeys;
  }
  return keys2;
}
function isUniquePrefixHash(altKnownPathsKeys, searchPathKeys, idx) {
  for (let currAltIdx = 0; currAltIdx < altKnownPathsKeys.length; currAltIdx++) {
    if (currAltIdx === idx) {
      continue;
    }
    const otherAltKnownPathsKeys = altKnownPathsKeys[currAltIdx];
    for (let searchIdx = 0; searchIdx < searchPathKeys.length; searchIdx++) {
      const searchKey = searchPathKeys[searchIdx];
      if (otherAltKnownPathsKeys[searchKey] === true) {
        return false;
      }
    }
  }
  return true;
}
function lookAheadSequenceFromAlternatives(altsDefs, k) {
  const partialAlts = map_default(altsDefs, (currAlt) => possiblePathsFrom([currAlt], 1));
  const finalResult = initializeArrayOfArrays(partialAlts.length);
  const altsHashes = map_default(partialAlts, (currAltPaths) => {
    const dict = {};
    forEach_default(currAltPaths, (item) => {
      const keys2 = pathToHashKeys(item.partialPath);
      forEach_default(keys2, (currKey) => {
        dict[currKey] = true;
      });
    });
    return dict;
  });
  let newData = partialAlts;
  for (let pathLength = 1; pathLength <= k; pathLength++) {
    const currDataset = newData;
    newData = initializeArrayOfArrays(currDataset.length);
    for (let altIdx = 0; altIdx < currDataset.length; altIdx++) {
      const currAltPathsAndSuffixes = currDataset[altIdx];
      for (let currPathIdx = 0; currPathIdx < currAltPathsAndSuffixes.length; currPathIdx++) {
        const currPathPrefix = currAltPathsAndSuffixes[currPathIdx].partialPath;
        const suffixDef = currAltPathsAndSuffixes[currPathIdx].suffixDef;
        const prefixKeys = pathToHashKeys(currPathPrefix);
        const isUnique = isUniquePrefixHash(altsHashes, prefixKeys, altIdx);
        if (isUnique || isEmpty_default(suffixDef) || currPathPrefix.length === k) {
          const currAltResult = finalResult[altIdx];
          if (containsPath(currAltResult, currPathPrefix) === false) {
            currAltResult.push(currPathPrefix);
            for (let j = 0; j < prefixKeys.length; j++) {
              const currKey = prefixKeys[j];
              altsHashes[altIdx][currKey] = true;
            }
          }
        } else {
          const newPartialPathsAndSuffixes = possiblePathsFrom(suffixDef, pathLength + 1, currPathPrefix);
          newData[altIdx] = newData[altIdx].concat(newPartialPathsAndSuffixes);
          forEach_default(newPartialPathsAndSuffixes, (item) => {
            const prefixKeys2 = pathToHashKeys(item.partialPath);
            forEach_default(prefixKeys2, (key) => {
              altsHashes[altIdx][key] = true;
            });
          });
        }
      }
    }
  }
  return finalResult;
}
function getLookaheadPathsForOr(occurrence, ruleGrammar, k, orProd) {
  const visitor2 = new InsideDefinitionFinderVisitor(occurrence, PROD_TYPE.ALTERNATION, orProd);
  ruleGrammar.accept(visitor2);
  return lookAheadSequenceFromAlternatives(visitor2.result, k);
}
function getLookaheadPathsForOptionalProd(occurrence, ruleGrammar, prodType, k) {
  const insideDefVisitor = new InsideDefinitionFinderVisitor(occurrence, prodType);
  ruleGrammar.accept(insideDefVisitor);
  const insideDef = insideDefVisitor.result;
  const afterDefWalker = new RestDefinitionFinderWalker(ruleGrammar, occurrence, prodType);
  const afterDef = afterDefWalker.startWalking();
  const insideFlat = new Alternative({ definition: insideDef });
  const afterFlat = new Alternative({ definition: afterDef });
  return lookAheadSequenceFromAlternatives([insideFlat, afterFlat], k);
}
function containsPath(alternative, searchPath) {
  compareOtherPath: for (let i = 0; i < alternative.length; i++) {
    const otherPath = alternative[i];
    if (otherPath.length !== searchPath.length) {
      continue;
    }
    for (let j = 0; j < otherPath.length; j++) {
      const searchTok = searchPath[j];
      const otherTok = otherPath[j];
      const matchingTokens = searchTok === otherTok || otherTok.categoryMatchesMap[searchTok.tokenTypeIdx] !== void 0;
      if (matchingTokens === false) {
        continue compareOtherPath;
      }
    }
    return true;
  }
  return false;
}
function isStrictPrefixOfPath(prefix, other) {
  return prefix.length < other.length && every_default(prefix, (tokType, idx) => {
    const otherTokType = other[idx];
    return tokType === otherTokType || otherTokType.categoryMatchesMap[tokType.tokenTypeIdx];
  });
}
function areTokenCategoriesNotUsed(lookAheadPaths) {
  return every_default(lookAheadPaths, (singleAltPaths) => every_default(singleAltPaths, (singlePath) => every_default(singlePath, (token) => isEmpty_default(token.categoryMatches))));
}

// node_modules/chevrotain/lib/src/parse/grammar/checks.js
function validateLookahead(options) {
  const lookaheadValidationErrorMessages = options.lookaheadStrategy.validate({
    rules: options.rules,
    tokenTypes: options.tokenTypes,
    grammarName: options.grammarName
  });
  return map_default(lookaheadValidationErrorMessages, (errorMessage) => Object.assign({ type: ParserDefinitionErrorType.CUSTOM_LOOKAHEAD_VALIDATION }, errorMessage));
}
function validateGrammar(topLevels, tokenTypes, errMsgProvider, grammarName) {
  const duplicateErrors = flatMap_default(topLevels, (currTopLevel) => validateDuplicateProductions(currTopLevel, errMsgProvider));
  const termsNamespaceConflictErrors = checkTerminalAndNoneTerminalsNameSpace(topLevels, tokenTypes, errMsgProvider);
  const tooManyAltsErrors = flatMap_default(topLevels, (curRule) => validateTooManyAlts(curRule, errMsgProvider));
  const duplicateRulesError = flatMap_default(topLevels, (curRule) => validateRuleDoesNotAlreadyExist(curRule, topLevels, grammarName, errMsgProvider));
  return duplicateErrors.concat(termsNamespaceConflictErrors, tooManyAltsErrors, duplicateRulesError);
}
function validateDuplicateProductions(topLevelRule, errMsgProvider) {
  const collectorVisitor2 = new OccurrenceValidationCollector();
  topLevelRule.accept(collectorVisitor2);
  const allRuleProductions = collectorVisitor2.allProductions;
  const productionGroups = groupBy_default(allRuleProductions, identifyProductionForDuplicates);
  const duplicates = pickBy_default(productionGroups, (currGroup) => {
    return currGroup.length > 1;
  });
  const errors = map_default(values_default(duplicates), (currDuplicates) => {
    const firstProd = head_default(currDuplicates);
    const msg = errMsgProvider.buildDuplicateFoundError(topLevelRule, currDuplicates);
    const dslName = getProductionDslName(firstProd);
    const defError = {
      message: msg,
      type: ParserDefinitionErrorType.DUPLICATE_PRODUCTIONS,
      ruleName: topLevelRule.name,
      dslName,
      occurrence: firstProd.idx
    };
    const param = getExtraProductionArgument(firstProd);
    if (param) {
      defError.parameter = param;
    }
    return defError;
  });
  return errors;
}
function identifyProductionForDuplicates(prod) {
  return `${getProductionDslName(prod)}_#_${prod.idx}_#_${getExtraProductionArgument(prod)}`;
}
function getExtraProductionArgument(prod) {
  if (prod instanceof Terminal) {
    return prod.terminalType.name;
  } else if (prod instanceof NonTerminal) {
    return prod.nonTerminalName;
  } else {
    return "";
  }
}
var OccurrenceValidationCollector = class extends GAstVisitor {
  constructor() {
    super(...arguments);
    this.allProductions = [];
  }
  visitNonTerminal(subrule) {
    this.allProductions.push(subrule);
  }
  visitOption(option2) {
    this.allProductions.push(option2);
  }
  visitRepetitionWithSeparator(manySep) {
    this.allProductions.push(manySep);
  }
  visitRepetitionMandatory(atLeastOne) {
    this.allProductions.push(atLeastOne);
  }
  visitRepetitionMandatoryWithSeparator(atLeastOneSep) {
    this.allProductions.push(atLeastOneSep);
  }
  visitRepetition(many) {
    this.allProductions.push(many);
  }
  visitAlternation(or) {
    this.allProductions.push(or);
  }
  visitTerminal(terminal) {
    this.allProductions.push(terminal);
  }
};
function validateRuleDoesNotAlreadyExist(rule, allRules, className, errMsgProvider) {
  const errors = [];
  const occurrences = reduce_default(allRules, (result2, curRule) => {
    if (curRule.name === rule.name) {
      return result2 + 1;
    }
    return result2;
  }, 0);
  if (occurrences > 1) {
    const errMsg = errMsgProvider.buildDuplicateRuleNameError({
      topLevelRule: rule,
      grammarName: className
    });
    errors.push({
      message: errMsg,
      type: ParserDefinitionErrorType.DUPLICATE_RULE_NAME,
      ruleName: rule.name
    });
  }
  return errors;
}
function validateRuleIsOverridden(ruleName, definedRulesNames, className) {
  const errors = [];
  let errMsg;
  if (!includes_default(definedRulesNames, ruleName)) {
    errMsg = `Invalid rule override, rule: ->${ruleName}<- cannot be overridden in the grammar: ->${className}<-as it is not defined in any of the super grammars `;
    errors.push({
      message: errMsg,
      type: ParserDefinitionErrorType.INVALID_RULE_OVERRIDE,
      ruleName
    });
  }
  return errors;
}
function validateNoLeftRecursion(topRule, currRule, errMsgProvider, path = []) {
  const errors = [];
  const nextNonTerminals = getFirstNoneTerminal(currRule.definition);
  if (isEmpty_default(nextNonTerminals)) {
    return [];
  } else {
    const ruleName = topRule.name;
    const foundLeftRecursion = includes_default(nextNonTerminals, topRule);
    if (foundLeftRecursion) {
      errors.push({
        message: errMsgProvider.buildLeftRecursionError({
          topLevelRule: topRule,
          leftRecursionPath: path
        }),
        type: ParserDefinitionErrorType.LEFT_RECURSION,
        ruleName
      });
    }
    const validNextSteps = difference_default(nextNonTerminals, path.concat([topRule]));
    const errorsFromNextSteps = flatMap_default(validNextSteps, (currRefRule) => {
      const newPath = clone_default(path);
      newPath.push(currRefRule);
      return validateNoLeftRecursion(topRule, currRefRule, errMsgProvider, newPath);
    });
    return errors.concat(errorsFromNextSteps);
  }
}
function getFirstNoneTerminal(definition) {
  let result2 = [];
  if (isEmpty_default(definition)) {
    return result2;
  }
  const firstProd = head_default(definition);
  if (firstProd instanceof NonTerminal) {
    result2.push(firstProd.referencedRule);
  } else if (firstProd instanceof Alternative || firstProd instanceof Option || firstProd instanceof RepetitionMandatory || firstProd instanceof RepetitionMandatoryWithSeparator || firstProd instanceof RepetitionWithSeparator || firstProd instanceof Repetition) {
    result2 = result2.concat(getFirstNoneTerminal(firstProd.definition));
  } else if (firstProd instanceof Alternation) {
    result2 = flatten_default(map_default(firstProd.definition, (currSubDef) => getFirstNoneTerminal(currSubDef.definition)));
  } else if (firstProd instanceof Terminal) {
  } else {
    throw Error("non exhaustive match");
  }
  const isFirstOptional = isOptionalProd(firstProd);
  const hasMore = definition.length > 1;
  if (isFirstOptional && hasMore) {
    const rest2 = drop_default(definition);
    return result2.concat(getFirstNoneTerminal(rest2));
  } else {
    return result2;
  }
}
var OrCollector = class extends GAstVisitor {
  constructor() {
    super(...arguments);
    this.alternations = [];
  }
  visitAlternation(node) {
    this.alternations.push(node);
  }
};
function validateEmptyOrAlternative(topLevelRule, errMsgProvider) {
  const orCollector = new OrCollector();
  topLevelRule.accept(orCollector);
  const ors = orCollector.alternations;
  const errors = flatMap_default(ors, (currOr) => {
    const exceptLast = dropRight_default(currOr.definition);
    return flatMap_default(exceptLast, (currAlternative, currAltIdx) => {
      const possibleFirstInAlt = nextPossibleTokensAfter([currAlternative], [], tokenStructuredMatcher, 1);
      if (isEmpty_default(possibleFirstInAlt)) {
        return [
          {
            message: errMsgProvider.buildEmptyAlternationError({
              topLevelRule,
              alternation: currOr,
              emptyChoiceIdx: currAltIdx
            }),
            type: ParserDefinitionErrorType.NONE_LAST_EMPTY_ALT,
            ruleName: topLevelRule.name,
            occurrence: currOr.idx,
            alternative: currAltIdx + 1
          }
        ];
      } else {
        return [];
      }
    });
  });
  return errors;
}
function validateAmbiguousAlternationAlternatives(topLevelRule, globalMaxLookahead, errMsgProvider) {
  const orCollector = new OrCollector();
  topLevelRule.accept(orCollector);
  let ors = orCollector.alternations;
  ors = reject_default(ors, (currOr) => currOr.ignoreAmbiguities === true);
  const errors = flatMap_default(ors, (currOr) => {
    const currOccurrence = currOr.idx;
    const actualMaxLookahead = currOr.maxLookahead || globalMaxLookahead;
    const alternatives = getLookaheadPathsForOr(currOccurrence, topLevelRule, actualMaxLookahead, currOr);
    const altsAmbiguityErrors = checkAlternativesAmbiguities(alternatives, currOr, topLevelRule, errMsgProvider);
    const altsPrefixAmbiguityErrors = checkPrefixAlternativesAmbiguities(alternatives, currOr, topLevelRule, errMsgProvider);
    return altsAmbiguityErrors.concat(altsPrefixAmbiguityErrors);
  });
  return errors;
}
var RepetitionCollector = class extends GAstVisitor {
  constructor() {
    super(...arguments);
    this.allProductions = [];
  }
  visitRepetitionWithSeparator(manySep) {
    this.allProductions.push(manySep);
  }
  visitRepetitionMandatory(atLeastOne) {
    this.allProductions.push(atLeastOne);
  }
  visitRepetitionMandatoryWithSeparator(atLeastOneSep) {
    this.allProductions.push(atLeastOneSep);
  }
  visitRepetition(many) {
    this.allProductions.push(many);
  }
};
function validateTooManyAlts(topLevelRule, errMsgProvider) {
  const orCollector = new OrCollector();
  topLevelRule.accept(orCollector);
  const ors = orCollector.alternations;
  const errors = flatMap_default(ors, (currOr) => {
    if (currOr.definition.length > 255) {
      return [
        {
          message: errMsgProvider.buildTooManyAlternativesError({
            topLevelRule,
            alternation: currOr
          }),
          type: ParserDefinitionErrorType.TOO_MANY_ALTS,
          ruleName: topLevelRule.name,
          occurrence: currOr.idx
        }
      ];
    } else {
      return [];
    }
  });
  return errors;
}
function validateSomeNonEmptyLookaheadPath(topLevelRules, maxLookahead, errMsgProvider) {
  const errors = [];
  forEach_default(topLevelRules, (currTopRule) => {
    const collectorVisitor2 = new RepetitionCollector();
    currTopRule.accept(collectorVisitor2);
    const allRuleProductions = collectorVisitor2.allProductions;
    forEach_default(allRuleProductions, (currProd) => {
      const prodType = getProdType(currProd);
      const actualMaxLookahead = currProd.maxLookahead || maxLookahead;
      const currOccurrence = currProd.idx;
      const paths = getLookaheadPathsForOptionalProd(currOccurrence, currTopRule, prodType, actualMaxLookahead);
      const pathsInsideProduction = paths[0];
      if (isEmpty_default(flatten_default(pathsInsideProduction))) {
        const errMsg = errMsgProvider.buildEmptyRepetitionError({
          topLevelRule: currTopRule,
          repetition: currProd
        });
        errors.push({
          message: errMsg,
          type: ParserDefinitionErrorType.NO_NON_EMPTY_LOOKAHEAD,
          ruleName: currTopRule.name
        });
      }
    });
  });
  return errors;
}
function checkAlternativesAmbiguities(alternatives, alternation2, rule, errMsgProvider) {
  const foundAmbiguousPaths = [];
  const identicalAmbiguities = reduce_default(alternatives, (result2, currAlt, currAltIdx) => {
    if (alternation2.definition[currAltIdx].ignoreAmbiguities === true) {
      return result2;
    }
    forEach_default(currAlt, (currPath) => {
      const altsCurrPathAppearsIn = [currAltIdx];
      forEach_default(alternatives, (currOtherAlt, currOtherAltIdx) => {
        if (currAltIdx !== currOtherAltIdx && containsPath(currOtherAlt, currPath) && // ignore (skip) ambiguities with this "other" alternative
        alternation2.definition[currOtherAltIdx].ignoreAmbiguities !== true) {
          altsCurrPathAppearsIn.push(currOtherAltIdx);
        }
      });
      if (altsCurrPathAppearsIn.length > 1 && !containsPath(foundAmbiguousPaths, currPath)) {
        foundAmbiguousPaths.push(currPath);
        result2.push({
          alts: altsCurrPathAppearsIn,
          path: currPath
        });
      }
    });
    return result2;
  }, []);
  const currErrors = map_default(identicalAmbiguities, (currAmbDescriptor) => {
    const ambgIndices = map_default(currAmbDescriptor.alts, (currAltIdx) => currAltIdx + 1);
    const currMessage = errMsgProvider.buildAlternationAmbiguityError({
      topLevelRule: rule,
      alternation: alternation2,
      ambiguityIndices: ambgIndices,
      prefixPath: currAmbDescriptor.path
    });
    return {
      message: currMessage,
      type: ParserDefinitionErrorType.AMBIGUOUS_ALTS,
      ruleName: rule.name,
      occurrence: alternation2.idx,
      alternatives: currAmbDescriptor.alts
    };
  });
  return currErrors;
}
function checkPrefixAlternativesAmbiguities(alternatives, alternation2, rule, errMsgProvider) {
  const pathsAndIndices = reduce_default(alternatives, (result2, currAlt, idx) => {
    const currPathsAndIdx = map_default(currAlt, (currPath) => {
      return { idx, path: currPath };
    });
    return result2.concat(currPathsAndIdx);
  }, []);
  const errors = compact_default(flatMap_default(pathsAndIndices, (currPathAndIdx) => {
    const alternativeGast = alternation2.definition[currPathAndIdx.idx];
    if (alternativeGast.ignoreAmbiguities === true) {
      return [];
    }
    const targetIdx = currPathAndIdx.idx;
    const targetPath = currPathAndIdx.path;
    const prefixAmbiguitiesPathsAndIndices = filter_default(pathsAndIndices, (searchPathAndIdx) => {
      return (
        // ignore (skip) ambiguities with this "other" alternative
        alternation2.definition[searchPathAndIdx.idx].ignoreAmbiguities !== true && searchPathAndIdx.idx < targetIdx && // checking for strict prefix because identical lookaheads
        // will be be detected using a different validation.
        isStrictPrefixOfPath(searchPathAndIdx.path, targetPath)
      );
    });
    const currPathPrefixErrors = map_default(prefixAmbiguitiesPathsAndIndices, (currAmbPathAndIdx) => {
      const ambgIndices = [currAmbPathAndIdx.idx + 1, targetIdx + 1];
      const occurrence = alternation2.idx === 0 ? "" : alternation2.idx;
      const message = errMsgProvider.buildAlternationPrefixAmbiguityError({
        topLevelRule: rule,
        alternation: alternation2,
        ambiguityIndices: ambgIndices,
        prefixPath: currAmbPathAndIdx.path
      });
      return {
        message,
        type: ParserDefinitionErrorType.AMBIGUOUS_PREFIX_ALTS,
        ruleName: rule.name,
        occurrence,
        alternatives: ambgIndices
      };
    });
    return currPathPrefixErrors;
  }));
  return errors;
}
function checkTerminalAndNoneTerminalsNameSpace(topLevels, tokenTypes, errMsgProvider) {
  const errors = [];
  const tokenNames = map_default(tokenTypes, (currToken) => currToken.name);
  forEach_default(topLevels, (currRule) => {
    const currRuleName = currRule.name;
    if (includes_default(tokenNames, currRuleName)) {
      const errMsg = errMsgProvider.buildNamespaceConflictError(currRule);
      errors.push({
        message: errMsg,
        type: ParserDefinitionErrorType.CONFLICT_TOKENS_RULES_NAMESPACE,
        ruleName: currRuleName
      });
    }
  });
  return errors;
}

// node_modules/chevrotain/lib/src/parse/grammar/gast/gast_resolver_public.js
function resolveGrammar2(options) {
  const actualOptions = defaults_default(options, {
    errMsgProvider: defaultGrammarResolverErrorProvider
  });
  const topRulesTable = {};
  forEach_default(options.rules, (rule) => {
    topRulesTable[rule.name] = rule;
  });
  return resolveGrammar(topRulesTable, actualOptions.errMsgProvider);
}
function validateGrammar2(options) {
  options = defaults_default(options, {
    errMsgProvider: defaultGrammarValidatorErrorProvider
  });
  return validateGrammar(options.rules, options.tokenTypes, options.errMsgProvider, options.grammarName);
}

// node_modules/chevrotain/lib/src/parse/exceptions_public.js
var MISMATCHED_TOKEN_EXCEPTION = "MismatchedTokenException";
var NO_VIABLE_ALT_EXCEPTION = "NoViableAltException";
var EARLY_EXIT_EXCEPTION = "EarlyExitException";
var NOT_ALL_INPUT_PARSED_EXCEPTION = "NotAllInputParsedException";
var RECOGNITION_EXCEPTION_NAMES = [
  MISMATCHED_TOKEN_EXCEPTION,
  NO_VIABLE_ALT_EXCEPTION,
  EARLY_EXIT_EXCEPTION,
  NOT_ALL_INPUT_PARSED_EXCEPTION
];
Object.freeze(RECOGNITION_EXCEPTION_NAMES);
function isRecognitionException(error) {
  return includes_default(RECOGNITION_EXCEPTION_NAMES, error.name);
}
var RecognitionException = class extends Error {
  constructor(message, token) {
    super(message);
    this.token = token;
    this.resyncedTokens = [];
    Object.setPrototypeOf(this, new.target.prototype);
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
};
var MismatchedTokenException = class extends RecognitionException {
  constructor(message, token, previousToken) {
    super(message, token);
    this.previousToken = previousToken;
    this.name = MISMATCHED_TOKEN_EXCEPTION;
  }
};
var NoViableAltException = class extends RecognitionException {
  constructor(message, token, previousToken) {
    super(message, token);
    this.previousToken = previousToken;
    this.name = NO_VIABLE_ALT_EXCEPTION;
  }
};
var NotAllInputParsedException = class extends RecognitionException {
  constructor(message, token) {
    super(message, token);
    this.name = NOT_ALL_INPUT_PARSED_EXCEPTION;
  }
};
var EarlyExitException = class extends RecognitionException {
  constructor(message, token, previousToken) {
    super(message, token);
    this.previousToken = previousToken;
    this.name = EARLY_EXIT_EXCEPTION;
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/recoverable.js
var EOF_FOLLOW_KEY = {};
var IN_RULE_RECOVERY_EXCEPTION = "InRuleRecoveryException";
var InRuleRecoveryException = class extends Error {
  constructor(message) {
    super(message);
    this.name = IN_RULE_RECOVERY_EXCEPTION;
  }
};
var Recoverable = class {
  initRecoverable(config) {
    this.firstAfterRepMap = {};
    this.resyncFollows = {};
    this.recoveryEnabled = has_default(config, "recoveryEnabled") ? config.recoveryEnabled : DEFAULT_PARSER_CONFIG.recoveryEnabled;
    if (this.recoveryEnabled) {
      this.attemptInRepetitionRecovery = attemptInRepetitionRecovery;
    }
  }
  getTokenToInsert(tokType) {
    const tokToInsert = createTokenInstance(tokType, "", NaN, NaN, NaN, NaN, NaN, NaN);
    tokToInsert.isInsertedInRecovery = true;
    return tokToInsert;
  }
  canTokenTypeBeInsertedInRecovery(tokType) {
    return true;
  }
  canTokenTypeBeDeletedInRecovery(tokType) {
    return true;
  }
  tryInRepetitionRecovery(grammarRule, grammarRuleArgs, lookAheadFunc, expectedTokType) {
    const reSyncTokType = this.findReSyncTokenType();
    const savedLexerState = this.exportLexerState();
    const resyncedTokens = [];
    let passedResyncPoint = false;
    const nextTokenWithoutResync = this.LA(1);
    let currToken = this.LA(1);
    const generateErrorMessage = () => {
      const previousToken = this.LA(0);
      const msg = this.errorMessageProvider.buildMismatchTokenMessage({
        expected: expectedTokType,
        actual: nextTokenWithoutResync,
        previous: previousToken,
        ruleName: this.getCurrRuleFullName()
      });
      const error = new MismatchedTokenException(msg, nextTokenWithoutResync, this.LA(0));
      error.resyncedTokens = dropRight_default(resyncedTokens);
      this.SAVE_ERROR(error);
    };
    while (!passedResyncPoint) {
      if (this.tokenMatcher(currToken, expectedTokType)) {
        generateErrorMessage();
        return;
      } else if (lookAheadFunc.call(this)) {
        generateErrorMessage();
        grammarRule.apply(this, grammarRuleArgs);
        return;
      } else if (this.tokenMatcher(currToken, reSyncTokType)) {
        passedResyncPoint = true;
      } else {
        currToken = this.SKIP_TOKEN();
        this.addToResyncTokens(currToken, resyncedTokens);
      }
    }
    this.importLexerState(savedLexerState);
  }
  shouldInRepetitionRecoveryBeTried(expectTokAfterLastMatch, nextTokIdx, notStuck) {
    if (notStuck === false) {
      return false;
    }
    if (this.tokenMatcher(this.LA(1), expectTokAfterLastMatch)) {
      return false;
    }
    if (this.isBackTracking()) {
      return false;
    }
    if (this.canPerformInRuleRecovery(expectTokAfterLastMatch, this.getFollowsForInRuleRecovery(expectTokAfterLastMatch, nextTokIdx))) {
      return false;
    }
    return true;
  }
  // Error Recovery functionality
  getFollowsForInRuleRecovery(tokType, tokIdxInRule) {
    const grammarPath = this.getCurrentGrammarPath(tokType, tokIdxInRule);
    const follows = this.getNextPossibleTokenTypes(grammarPath);
    return follows;
  }
  tryInRuleRecovery(expectedTokType, follows) {
    if (this.canRecoverWithSingleTokenInsertion(expectedTokType, follows)) {
      const tokToInsert = this.getTokenToInsert(expectedTokType);
      return tokToInsert;
    }
    if (this.canRecoverWithSingleTokenDeletion(expectedTokType)) {
      const nextTok = this.SKIP_TOKEN();
      this.consumeToken();
      return nextTok;
    }
    throw new InRuleRecoveryException("sad sad panda");
  }
  canPerformInRuleRecovery(expectedToken, follows) {
    return this.canRecoverWithSingleTokenInsertion(expectedToken, follows) || this.canRecoverWithSingleTokenDeletion(expectedToken);
  }
  canRecoverWithSingleTokenInsertion(expectedTokType, follows) {
    if (!this.canTokenTypeBeInsertedInRecovery(expectedTokType)) {
      return false;
    }
    if (isEmpty_default(follows)) {
      return false;
    }
    const mismatchedTok = this.LA(1);
    const isMisMatchedTokInFollows = find_default(follows, (possibleFollowsTokType) => {
      return this.tokenMatcher(mismatchedTok, possibleFollowsTokType);
    }) !== void 0;
    return isMisMatchedTokInFollows;
  }
  canRecoverWithSingleTokenDeletion(expectedTokType) {
    if (!this.canTokenTypeBeDeletedInRecovery(expectedTokType)) {
      return false;
    }
    const isNextTokenWhatIsExpected = this.tokenMatcher(this.LA(2), expectedTokType);
    return isNextTokenWhatIsExpected;
  }
  isInCurrentRuleReSyncSet(tokenTypeIdx) {
    const followKey = this.getCurrFollowKey();
    const currentRuleReSyncSet = this.getFollowSetFromFollowKey(followKey);
    return includes_default(currentRuleReSyncSet, tokenTypeIdx);
  }
  findReSyncTokenType() {
    const allPossibleReSyncTokTypes = this.flattenFollowSet();
    let nextToken = this.LA(1);
    let k = 2;
    while (true) {
      const foundMatch = find_default(allPossibleReSyncTokTypes, (resyncTokType) => {
        const canMatch = tokenMatcher(nextToken, resyncTokType);
        return canMatch;
      });
      if (foundMatch !== void 0) {
        return foundMatch;
      }
      nextToken = this.LA(k);
      k++;
    }
  }
  getCurrFollowKey() {
    if (this.RULE_STACK.length === 1) {
      return EOF_FOLLOW_KEY;
    }
    const currRuleShortName = this.getLastExplicitRuleShortName();
    const currRuleIdx = this.getLastExplicitRuleOccurrenceIndex();
    const prevRuleShortName = this.getPreviousExplicitRuleShortName();
    return {
      ruleName: this.shortRuleNameToFullName(currRuleShortName),
      idxInCallingRule: currRuleIdx,
      inRule: this.shortRuleNameToFullName(prevRuleShortName)
    };
  }
  buildFullFollowKeyStack() {
    const explicitRuleStack = this.RULE_STACK;
    const explicitOccurrenceStack = this.RULE_OCCURRENCE_STACK;
    return map_default(explicitRuleStack, (ruleName, idx) => {
      if (idx === 0) {
        return EOF_FOLLOW_KEY;
      }
      return {
        ruleName: this.shortRuleNameToFullName(ruleName),
        idxInCallingRule: explicitOccurrenceStack[idx],
        inRule: this.shortRuleNameToFullName(explicitRuleStack[idx - 1])
      };
    });
  }
  flattenFollowSet() {
    const followStack = map_default(this.buildFullFollowKeyStack(), (currKey) => {
      return this.getFollowSetFromFollowKey(currKey);
    });
    return flatten_default(followStack);
  }
  getFollowSetFromFollowKey(followKey) {
    if (followKey === EOF_FOLLOW_KEY) {
      return [EOF];
    }
    const followName = followKey.ruleName + followKey.idxInCallingRule + IN + followKey.inRule;
    return this.resyncFollows[followName];
  }
  // It does not make any sense to include a virtual EOF token in the list of resynced tokens
  // as EOF does not really exist and thus does not contain any useful information (line/column numbers)
  addToResyncTokens(token, resyncTokens) {
    if (!this.tokenMatcher(token, EOF)) {
      resyncTokens.push(token);
    }
    return resyncTokens;
  }
  reSyncTo(tokType) {
    const resyncedTokens = [];
    let nextTok = this.LA(1);
    while (this.tokenMatcher(nextTok, tokType) === false) {
      nextTok = this.SKIP_TOKEN();
      this.addToResyncTokens(nextTok, resyncedTokens);
    }
    return dropRight_default(resyncedTokens);
  }
  attemptInRepetitionRecovery(prodFunc, args, lookaheadFunc, dslMethodIdx, prodOccurrence, nextToksWalker, notStuck) {
  }
  getCurrentGrammarPath(tokType, tokIdxInRule) {
    const pathRuleStack = this.getHumanReadableRuleStack();
    const pathOccurrenceStack = clone_default(this.RULE_OCCURRENCE_STACK);
    const grammarPath = {
      ruleStack: pathRuleStack,
      occurrenceStack: pathOccurrenceStack,
      lastTok: tokType,
      lastTokOccurrence: tokIdxInRule
    };
    return grammarPath;
  }
  getHumanReadableRuleStack() {
    return map_default(this.RULE_STACK, (currShortName) => this.shortRuleNameToFullName(currShortName));
  }
};
function attemptInRepetitionRecovery(prodFunc, args, lookaheadFunc, dslMethodIdx, prodOccurrence, nextToksWalker, notStuck) {
  const key = this.getKeyForAutomaticLookahead(dslMethodIdx, prodOccurrence);
  let firstAfterRepInfo = this.firstAfterRepMap[key];
  if (firstAfterRepInfo === void 0) {
    const currRuleName = this.getCurrRuleFullName();
    const ruleGrammar = this.getGAstProductions()[currRuleName];
    const walker = new nextToksWalker(ruleGrammar, prodOccurrence);
    firstAfterRepInfo = walker.startWalking();
    this.firstAfterRepMap[key] = firstAfterRepInfo;
  }
  let expectTokAfterLastMatch = firstAfterRepInfo.token;
  let nextTokIdx = firstAfterRepInfo.occurrence;
  const isEndOfRule = firstAfterRepInfo.isEndOfRule;
  if (this.RULE_STACK.length === 1 && isEndOfRule && expectTokAfterLastMatch === void 0) {
    expectTokAfterLastMatch = EOF;
    nextTokIdx = 1;
  }
  if (expectTokAfterLastMatch === void 0 || nextTokIdx === void 0) {
    return;
  }
  if (this.shouldInRepetitionRecoveryBeTried(expectTokAfterLastMatch, nextTokIdx, notStuck)) {
    this.tryInRepetitionRecovery(prodFunc, args, lookaheadFunc, expectTokAfterLastMatch);
  }
}

// node_modules/chevrotain/lib/src/parse/grammar/keys.js
var BITS_FOR_METHOD_TYPE = 4;
var BITS_FOR_OCCURRENCE_IDX = 8;
var BITS_FOR_ALT_IDX = 8;
var OR_IDX = 1 << BITS_FOR_OCCURRENCE_IDX;
var OPTION_IDX = 2 << BITS_FOR_OCCURRENCE_IDX;
var MANY_IDX = 3 << BITS_FOR_OCCURRENCE_IDX;
var AT_LEAST_ONE_IDX = 4 << BITS_FOR_OCCURRENCE_IDX;
var MANY_SEP_IDX = 5 << BITS_FOR_OCCURRENCE_IDX;
var AT_LEAST_ONE_SEP_IDX = 6 << BITS_FOR_OCCURRENCE_IDX;
function getKeyForAutomaticLookahead(ruleIdx, dslMethodIdx, occurrence) {
  return occurrence | dslMethodIdx | ruleIdx;
}
var BITS_START_FOR_ALT_IDX = 32 - BITS_FOR_ALT_IDX;

// node_modules/chevrotain/lib/src/parse/grammar/llk_lookahead.js
var LLkLookaheadStrategy = class {
  constructor(options) {
    var _a;
    this.maxLookahead = (_a = options === null || options === void 0 ? void 0 : options.maxLookahead) !== null && _a !== void 0 ? _a : DEFAULT_PARSER_CONFIG.maxLookahead;
  }
  validate(options) {
    const leftRecursionErrors = this.validateNoLeftRecursion(options.rules);
    if (isEmpty_default(leftRecursionErrors)) {
      const emptyAltErrors = this.validateEmptyOrAlternatives(options.rules);
      const ambiguousAltsErrors = this.validateAmbiguousAlternationAlternatives(options.rules, this.maxLookahead);
      const emptyRepetitionErrors = this.validateSomeNonEmptyLookaheadPath(options.rules, this.maxLookahead);
      const allErrors = [
        ...leftRecursionErrors,
        ...emptyAltErrors,
        ...ambiguousAltsErrors,
        ...emptyRepetitionErrors
      ];
      return allErrors;
    }
    return leftRecursionErrors;
  }
  validateNoLeftRecursion(rules) {
    return flatMap_default(rules, (currTopRule) => validateNoLeftRecursion(currTopRule, currTopRule, defaultGrammarValidatorErrorProvider));
  }
  validateEmptyOrAlternatives(rules) {
    return flatMap_default(rules, (currTopRule) => validateEmptyOrAlternative(currTopRule, defaultGrammarValidatorErrorProvider));
  }
  validateAmbiguousAlternationAlternatives(rules, maxLookahead) {
    return flatMap_default(rules, (currTopRule) => validateAmbiguousAlternationAlternatives(currTopRule, maxLookahead, defaultGrammarValidatorErrorProvider));
  }
  validateSomeNonEmptyLookaheadPath(rules, maxLookahead) {
    return validateSomeNonEmptyLookaheadPath(rules, maxLookahead, defaultGrammarValidatorErrorProvider);
  }
  buildLookaheadForAlternation(options) {
    return buildLookaheadFuncForOr(options.prodOccurrence, options.rule, options.maxLookahead, options.hasPredicates, options.dynamicTokensEnabled, buildAlternativesLookAheadFunc);
  }
  buildLookaheadForOptional(options) {
    return buildLookaheadFuncForOptionalProd(options.prodOccurrence, options.rule, options.maxLookahead, options.dynamicTokensEnabled, getProdType(options.prodType), buildSingleAlternativeLookaheadFunction);
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/looksahead.js
var LooksAhead = class {
  initLooksAhead(config) {
    this.dynamicTokensEnabled = has_default(config, "dynamicTokensEnabled") ? config.dynamicTokensEnabled : DEFAULT_PARSER_CONFIG.dynamicTokensEnabled;
    this.maxLookahead = has_default(config, "maxLookahead") ? config.maxLookahead : DEFAULT_PARSER_CONFIG.maxLookahead;
    this.lookaheadStrategy = has_default(config, "lookaheadStrategy") ? config.lookaheadStrategy : new LLkLookaheadStrategy({ maxLookahead: this.maxLookahead });
    this.lookAheadFuncsCache = /* @__PURE__ */ new Map();
  }
  preComputeLookaheadFunctions(rules) {
    forEach_default(rules, (currRule) => {
      this.TRACE_INIT(`${currRule.name} Rule Lookahead`, () => {
        const { alternation: alternation2, repetition: repetition2, option: option2, repetitionMandatory: repetitionMandatory2, repetitionMandatoryWithSeparator, repetitionWithSeparator } = collectMethods(currRule);
        forEach_default(alternation2, (currProd) => {
          const prodIdx = currProd.idx === 0 ? "" : currProd.idx;
          this.TRACE_INIT(`${getProductionDslName(currProd)}${prodIdx}`, () => {
            const laFunc = this.lookaheadStrategy.buildLookaheadForAlternation({
              prodOccurrence: currProd.idx,
              rule: currRule,
              maxLookahead: currProd.maxLookahead || this.maxLookahead,
              hasPredicates: currProd.hasPredicates,
              dynamicTokensEnabled: this.dynamicTokensEnabled
            });
            const key = getKeyForAutomaticLookahead(this.fullRuleNameToShort[currRule.name], OR_IDX, currProd.idx);
            this.setLaFuncCache(key, laFunc);
          });
        });
        forEach_default(repetition2, (currProd) => {
          this.computeLookaheadFunc(currRule, currProd.idx, MANY_IDX, "Repetition", currProd.maxLookahead, getProductionDslName(currProd));
        });
        forEach_default(option2, (currProd) => {
          this.computeLookaheadFunc(currRule, currProd.idx, OPTION_IDX, "Option", currProd.maxLookahead, getProductionDslName(currProd));
        });
        forEach_default(repetitionMandatory2, (currProd) => {
          this.computeLookaheadFunc(currRule, currProd.idx, AT_LEAST_ONE_IDX, "RepetitionMandatory", currProd.maxLookahead, getProductionDslName(currProd));
        });
        forEach_default(repetitionMandatoryWithSeparator, (currProd) => {
          this.computeLookaheadFunc(currRule, currProd.idx, AT_LEAST_ONE_SEP_IDX, "RepetitionMandatoryWithSeparator", currProd.maxLookahead, getProductionDslName(currProd));
        });
        forEach_default(repetitionWithSeparator, (currProd) => {
          this.computeLookaheadFunc(currRule, currProd.idx, MANY_SEP_IDX, "RepetitionWithSeparator", currProd.maxLookahead, getProductionDslName(currProd));
        });
      });
    });
  }
  computeLookaheadFunc(rule, prodOccurrence, prodKey, prodType, prodMaxLookahead, dslMethodName) {
    this.TRACE_INIT(`${dslMethodName}${prodOccurrence === 0 ? "" : prodOccurrence}`, () => {
      const laFunc = this.lookaheadStrategy.buildLookaheadForOptional({
        prodOccurrence,
        rule,
        maxLookahead: prodMaxLookahead || this.maxLookahead,
        dynamicTokensEnabled: this.dynamicTokensEnabled,
        prodType
      });
      const key = getKeyForAutomaticLookahead(this.fullRuleNameToShort[rule.name], prodKey, prodOccurrence);
      this.setLaFuncCache(key, laFunc);
    });
  }
  // this actually returns a number, but it is always used as a string (object prop key)
  getKeyForAutomaticLookahead(dslMethodIdx, occurrence) {
    const currRuleShortName = this.getLastExplicitRuleShortName();
    return getKeyForAutomaticLookahead(currRuleShortName, dslMethodIdx, occurrence);
  }
  getLaFuncFromCache(key) {
    return this.lookAheadFuncsCache.get(key);
  }
  /* istanbul ignore next */
  setLaFuncCache(key, value) {
    this.lookAheadFuncsCache.set(key, value);
  }
};
var DslMethodsCollectorVisitor = class extends GAstVisitor {
  constructor() {
    super(...arguments);
    this.dslMethods = {
      option: [],
      alternation: [],
      repetition: [],
      repetitionWithSeparator: [],
      repetitionMandatory: [],
      repetitionMandatoryWithSeparator: []
    };
  }
  reset() {
    this.dslMethods = {
      option: [],
      alternation: [],
      repetition: [],
      repetitionWithSeparator: [],
      repetitionMandatory: [],
      repetitionMandatoryWithSeparator: []
    };
  }
  visitOption(option2) {
    this.dslMethods.option.push(option2);
  }
  visitRepetitionWithSeparator(manySep) {
    this.dslMethods.repetitionWithSeparator.push(manySep);
  }
  visitRepetitionMandatory(atLeastOne) {
    this.dslMethods.repetitionMandatory.push(atLeastOne);
  }
  visitRepetitionMandatoryWithSeparator(atLeastOneSep) {
    this.dslMethods.repetitionMandatoryWithSeparator.push(atLeastOneSep);
  }
  visitRepetition(many) {
    this.dslMethods.repetition.push(many);
  }
  visitAlternation(or) {
    this.dslMethods.alternation.push(or);
  }
};
var collectorVisitor = new DslMethodsCollectorVisitor();
function collectMethods(rule) {
  collectorVisitor.reset();
  rule.accept(collectorVisitor);
  const dslMethods = collectorVisitor.dslMethods;
  collectorVisitor.reset();
  return dslMethods;
}

// node_modules/chevrotain/lib/src/parse/cst/cst.js
function setNodeLocationOnlyOffset(currNodeLocation, newLocationInfo) {
  if (isNaN(currNodeLocation.startOffset) === true) {
    currNodeLocation.startOffset = newLocationInfo.startOffset;
    currNodeLocation.endOffset = newLocationInfo.endOffset;
  } else if (currNodeLocation.endOffset < newLocationInfo.endOffset === true) {
    currNodeLocation.endOffset = newLocationInfo.endOffset;
  }
}
function setNodeLocationFull(currNodeLocation, newLocationInfo) {
  if (isNaN(currNodeLocation.startOffset) === true) {
    currNodeLocation.startOffset = newLocationInfo.startOffset;
    currNodeLocation.startColumn = newLocationInfo.startColumn;
    currNodeLocation.startLine = newLocationInfo.startLine;
    currNodeLocation.endOffset = newLocationInfo.endOffset;
    currNodeLocation.endColumn = newLocationInfo.endColumn;
    currNodeLocation.endLine = newLocationInfo.endLine;
  } else if (currNodeLocation.endOffset < newLocationInfo.endOffset === true) {
    currNodeLocation.endOffset = newLocationInfo.endOffset;
    currNodeLocation.endColumn = newLocationInfo.endColumn;
    currNodeLocation.endLine = newLocationInfo.endLine;
  }
}
function addTerminalToCst(node, token, tokenTypeName) {
  if (node.children[tokenTypeName] === void 0) {
    node.children[tokenTypeName] = [token];
  } else {
    node.children[tokenTypeName].push(token);
  }
}
function addNoneTerminalToCst(node, ruleName, ruleResult) {
  if (node.children[ruleName] === void 0) {
    node.children[ruleName] = [ruleResult];
  } else {
    node.children[ruleName].push(ruleResult);
  }
}

// node_modules/chevrotain/lib/src/lang/lang_extensions.js
var NAME = "name";
function defineNameProp(obj, nameValue) {
  Object.defineProperty(obj, NAME, {
    enumerable: false,
    configurable: true,
    writable: false,
    value: nameValue
  });
}

// node_modules/chevrotain/lib/src/parse/cst/cst_visitor.js
function defaultVisit(ctx, param) {
  const childrenNames = keys_default(ctx);
  const childrenNamesLength = childrenNames.length;
  for (let i = 0; i < childrenNamesLength; i++) {
    const currChildName = childrenNames[i];
    const currChildArray = ctx[currChildName];
    const currChildArrayLength = currChildArray.length;
    for (let j = 0; j < currChildArrayLength; j++) {
      const currChild = currChildArray[j];
      if (currChild.tokenTypeIdx === void 0) {
        this[currChild.name](currChild.children, param);
      }
    }
  }
}
function createBaseSemanticVisitorConstructor(grammarName, ruleNames) {
  const derivedConstructor = function() {
  };
  defineNameProp(derivedConstructor, grammarName + "BaseSemantics");
  const semanticProto = {
    visit: function(cstNode, param) {
      if (isArray_default(cstNode)) {
        cstNode = cstNode[0];
      }
      if (isUndefined_default(cstNode)) {
        return void 0;
      }
      return this[cstNode.name](cstNode.children, param);
    },
    validateVisitor: function() {
      const semanticDefinitionErrors = validateVisitor(this, ruleNames);
      if (!isEmpty_default(semanticDefinitionErrors)) {
        const errorMessages = map_default(semanticDefinitionErrors, (currDefError) => currDefError.msg);
        throw Error(`Errors Detected in CST Visitor <${this.constructor.name}>:
	${errorMessages.join("\n\n").replace(/\n/g, "\n	")}`);
      }
    }
  };
  derivedConstructor.prototype = semanticProto;
  derivedConstructor.prototype.constructor = derivedConstructor;
  derivedConstructor._RULE_NAMES = ruleNames;
  return derivedConstructor;
}
function createBaseVisitorConstructorWithDefaults(grammarName, ruleNames, baseConstructor) {
  const derivedConstructor = function() {
  };
  defineNameProp(derivedConstructor, grammarName + "BaseSemanticsWithDefaults");
  const withDefaultsProto = Object.create(baseConstructor.prototype);
  forEach_default(ruleNames, (ruleName) => {
    withDefaultsProto[ruleName] = defaultVisit;
  });
  derivedConstructor.prototype = withDefaultsProto;
  derivedConstructor.prototype.constructor = derivedConstructor;
  return derivedConstructor;
}
var CstVisitorDefinitionError;
(function(CstVisitorDefinitionError2) {
  CstVisitorDefinitionError2[CstVisitorDefinitionError2["REDUNDANT_METHOD"] = 0] = "REDUNDANT_METHOD";
  CstVisitorDefinitionError2[CstVisitorDefinitionError2["MISSING_METHOD"] = 1] = "MISSING_METHOD";
})(CstVisitorDefinitionError || (CstVisitorDefinitionError = {}));
function validateVisitor(visitorInstance, ruleNames) {
  const missingErrors = validateMissingCstMethods(visitorInstance, ruleNames);
  return missingErrors;
}
function validateMissingCstMethods(visitorInstance, ruleNames) {
  const missingRuleNames = filter_default(ruleNames, (currRuleName) => {
    return isFunction_default(visitorInstance[currRuleName]) === false;
  });
  const errors = map_default(missingRuleNames, (currRuleName) => {
    return {
      msg: `Missing visitor method: <${currRuleName}> on ${visitorInstance.constructor.name} CST Visitor.`,
      type: CstVisitorDefinitionError.MISSING_METHOD,
      methodName: currRuleName
    };
  });
  return compact_default(errors);
}

// node_modules/chevrotain/lib/src/parse/parser/traits/tree_builder.js
var TreeBuilder = class {
  initTreeBuilder(config) {
    this.CST_STACK = [];
    this.outputCst = config.outputCst;
    this.nodeLocationTracking = has_default(config, "nodeLocationTracking") ? config.nodeLocationTracking : DEFAULT_PARSER_CONFIG.nodeLocationTracking;
    if (!this.outputCst) {
      this.cstInvocationStateUpdate = noop_default;
      this.cstFinallyStateUpdate = noop_default;
      this.cstPostTerminal = noop_default;
      this.cstPostNonTerminal = noop_default;
      this.cstPostRule = noop_default;
    } else {
      if (/full/i.test(this.nodeLocationTracking)) {
        if (this.recoveryEnabled) {
          this.setNodeLocationFromToken = setNodeLocationFull;
          this.setNodeLocationFromNode = setNodeLocationFull;
          this.cstPostRule = noop_default;
          this.setInitialNodeLocation = this.setInitialNodeLocationFullRecovery;
        } else {
          this.setNodeLocationFromToken = noop_default;
          this.setNodeLocationFromNode = noop_default;
          this.cstPostRule = this.cstPostRuleFull;
          this.setInitialNodeLocation = this.setInitialNodeLocationFullRegular;
        }
      } else if (/onlyOffset/i.test(this.nodeLocationTracking)) {
        if (this.recoveryEnabled) {
          this.setNodeLocationFromToken = setNodeLocationOnlyOffset;
          this.setNodeLocationFromNode = setNodeLocationOnlyOffset;
          this.cstPostRule = noop_default;
          this.setInitialNodeLocation = this.setInitialNodeLocationOnlyOffsetRecovery;
        } else {
          this.setNodeLocationFromToken = noop_default;
          this.setNodeLocationFromNode = noop_default;
          this.cstPostRule = this.cstPostRuleOnlyOffset;
          this.setInitialNodeLocation = this.setInitialNodeLocationOnlyOffsetRegular;
        }
      } else if (/none/i.test(this.nodeLocationTracking)) {
        this.setNodeLocationFromToken = noop_default;
        this.setNodeLocationFromNode = noop_default;
        this.cstPostRule = noop_default;
        this.setInitialNodeLocation = noop_default;
      } else {
        throw Error(`Invalid <nodeLocationTracking> config option: "${config.nodeLocationTracking}"`);
      }
    }
  }
  setInitialNodeLocationOnlyOffsetRecovery(cstNode) {
    cstNode.location = {
      startOffset: NaN,
      endOffset: NaN
    };
  }
  setInitialNodeLocationOnlyOffsetRegular(cstNode) {
    cstNode.location = {
      // without error recovery the starting Location of a new CstNode is guaranteed
      // To be the next Token's startOffset (for valid inputs).
      // For invalid inputs there won't be any CSTOutput so this potential
      // inaccuracy does not matter
      startOffset: this.LA(1).startOffset,
      endOffset: NaN
    };
  }
  setInitialNodeLocationFullRecovery(cstNode) {
    cstNode.location = {
      startOffset: NaN,
      startLine: NaN,
      startColumn: NaN,
      endOffset: NaN,
      endLine: NaN,
      endColumn: NaN
    };
  }
  /**
       *  @see setInitialNodeLocationOnlyOffsetRegular for explanation why this work
  
       * @param cstNode
       */
  setInitialNodeLocationFullRegular(cstNode) {
    const nextToken = this.LA(1);
    cstNode.location = {
      startOffset: nextToken.startOffset,
      startLine: nextToken.startLine,
      startColumn: nextToken.startColumn,
      endOffset: NaN,
      endLine: NaN,
      endColumn: NaN
    };
  }
  cstInvocationStateUpdate(fullRuleName) {
    const cstNode = {
      name: fullRuleName,
      children: /* @__PURE__ */ Object.create(null)
    };
    this.setInitialNodeLocation(cstNode);
    this.CST_STACK.push(cstNode);
  }
  cstFinallyStateUpdate() {
    this.CST_STACK.pop();
  }
  cstPostRuleFull(ruleCstNode) {
    const prevToken = this.LA(0);
    const loc = ruleCstNode.location;
    if (loc.startOffset <= prevToken.startOffset === true) {
      loc.endOffset = prevToken.endOffset;
      loc.endLine = prevToken.endLine;
      loc.endColumn = prevToken.endColumn;
    } else {
      loc.startOffset = NaN;
      loc.startLine = NaN;
      loc.startColumn = NaN;
    }
  }
  cstPostRuleOnlyOffset(ruleCstNode) {
    const prevToken = this.LA(0);
    const loc = ruleCstNode.location;
    if (loc.startOffset <= prevToken.startOffset === true) {
      loc.endOffset = prevToken.endOffset;
    } else {
      loc.startOffset = NaN;
    }
  }
  cstPostTerminal(key, consumedToken) {
    const rootCst = this.CST_STACK[this.CST_STACK.length - 1];
    addTerminalToCst(rootCst, consumedToken, key);
    this.setNodeLocationFromToken(rootCst.location, consumedToken);
  }
  cstPostNonTerminal(ruleCstResult, ruleName) {
    const preCstNode = this.CST_STACK[this.CST_STACK.length - 1];
    addNoneTerminalToCst(preCstNode, ruleName, ruleCstResult);
    this.setNodeLocationFromNode(preCstNode.location, ruleCstResult.location);
  }
  getBaseCstVisitorConstructor() {
    if (isUndefined_default(this.baseCstVisitorConstructor)) {
      const newBaseCstVisitorConstructor = createBaseSemanticVisitorConstructor(this.className, keys_default(this.gastProductionsCache));
      this.baseCstVisitorConstructor = newBaseCstVisitorConstructor;
      return newBaseCstVisitorConstructor;
    }
    return this.baseCstVisitorConstructor;
  }
  getBaseCstVisitorConstructorWithDefaults() {
    if (isUndefined_default(this.baseCstVisitorWithDefaultsConstructor)) {
      const newConstructor = createBaseVisitorConstructorWithDefaults(this.className, keys_default(this.gastProductionsCache), this.getBaseCstVisitorConstructor());
      this.baseCstVisitorWithDefaultsConstructor = newConstructor;
      return newConstructor;
    }
    return this.baseCstVisitorWithDefaultsConstructor;
  }
  getLastExplicitRuleShortName() {
    const ruleStack = this.RULE_STACK;
    return ruleStack[ruleStack.length - 1];
  }
  getPreviousExplicitRuleShortName() {
    const ruleStack = this.RULE_STACK;
    return ruleStack[ruleStack.length - 2];
  }
  getLastExplicitRuleOccurrenceIndex() {
    const occurrenceStack = this.RULE_OCCURRENCE_STACK;
    return occurrenceStack[occurrenceStack.length - 1];
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/lexer_adapter.js
var LexerAdapter = class {
  initLexerAdapter() {
    this.tokVector = [];
    this.tokVectorLength = 0;
    this.currIdx = -1;
  }
  set input(newInput) {
    if (this.selfAnalysisDone !== true) {
      throw Error(`Missing <performSelfAnalysis> invocation at the end of the Parser's constructor.`);
    }
    this.reset();
    this.tokVector = newInput;
    this.tokVectorLength = newInput.length;
  }
  get input() {
    return this.tokVector;
  }
  // skips a token and returns the next token
  SKIP_TOKEN() {
    if (this.currIdx <= this.tokVector.length - 2) {
      this.consumeToken();
      return this.LA(1);
    } else {
      return END_OF_FILE;
    }
  }
  // Lexer (accessing Token vector) related methods which can be overridden to implement lazy lexers
  // or lexers dependent on parser context.
  LA(howMuch) {
    const soughtIdx = this.currIdx + howMuch;
    if (soughtIdx < 0 || this.tokVectorLength <= soughtIdx) {
      return END_OF_FILE;
    } else {
      return this.tokVector[soughtIdx];
    }
  }
  consumeToken() {
    this.currIdx++;
  }
  exportLexerState() {
    return this.currIdx;
  }
  importLexerState(newState2) {
    this.currIdx = newState2;
  }
  resetLexerState() {
    this.currIdx = -1;
  }
  moveToTerminatedState() {
    this.currIdx = this.tokVector.length - 1;
  }
  getLexerPosition() {
    return this.exportLexerState();
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/recognizer_api.js
var RecognizerApi = class {
  ACTION(impl) {
    return impl.call(this);
  }
  consume(idx, tokType, options) {
    return this.consumeInternal(tokType, idx, options);
  }
  subrule(idx, ruleToCall, options) {
    return this.subruleInternal(ruleToCall, idx, options);
  }
  option(idx, actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, idx);
  }
  or(idx, altsOrOpts) {
    return this.orInternal(altsOrOpts, idx);
  }
  many(idx, actionORMethodDef) {
    return this.manyInternal(idx, actionORMethodDef);
  }
  atLeastOne(idx, actionORMethodDef) {
    return this.atLeastOneInternal(idx, actionORMethodDef);
  }
  CONSUME(tokType, options) {
    return this.consumeInternal(tokType, 0, options);
  }
  CONSUME1(tokType, options) {
    return this.consumeInternal(tokType, 1, options);
  }
  CONSUME2(tokType, options) {
    return this.consumeInternal(tokType, 2, options);
  }
  CONSUME3(tokType, options) {
    return this.consumeInternal(tokType, 3, options);
  }
  CONSUME4(tokType, options) {
    return this.consumeInternal(tokType, 4, options);
  }
  CONSUME5(tokType, options) {
    return this.consumeInternal(tokType, 5, options);
  }
  CONSUME6(tokType, options) {
    return this.consumeInternal(tokType, 6, options);
  }
  CONSUME7(tokType, options) {
    return this.consumeInternal(tokType, 7, options);
  }
  CONSUME8(tokType, options) {
    return this.consumeInternal(tokType, 8, options);
  }
  CONSUME9(tokType, options) {
    return this.consumeInternal(tokType, 9, options);
  }
  SUBRULE(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 0, options);
  }
  SUBRULE1(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 1, options);
  }
  SUBRULE2(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 2, options);
  }
  SUBRULE3(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 3, options);
  }
  SUBRULE4(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 4, options);
  }
  SUBRULE5(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 5, options);
  }
  SUBRULE6(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 6, options);
  }
  SUBRULE7(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 7, options);
  }
  SUBRULE8(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 8, options);
  }
  SUBRULE9(ruleToCall, options) {
    return this.subruleInternal(ruleToCall, 9, options);
  }
  OPTION(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 0);
  }
  OPTION1(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 1);
  }
  OPTION2(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 2);
  }
  OPTION3(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 3);
  }
  OPTION4(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 4);
  }
  OPTION5(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 5);
  }
  OPTION6(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 6);
  }
  OPTION7(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 7);
  }
  OPTION8(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 8);
  }
  OPTION9(actionORMethodDef) {
    return this.optionInternal(actionORMethodDef, 9);
  }
  OR(altsOrOpts) {
    return this.orInternal(altsOrOpts, 0);
  }
  OR1(altsOrOpts) {
    return this.orInternal(altsOrOpts, 1);
  }
  OR2(altsOrOpts) {
    return this.orInternal(altsOrOpts, 2);
  }
  OR3(altsOrOpts) {
    return this.orInternal(altsOrOpts, 3);
  }
  OR4(altsOrOpts) {
    return this.orInternal(altsOrOpts, 4);
  }
  OR5(altsOrOpts) {
    return this.orInternal(altsOrOpts, 5);
  }
  OR6(altsOrOpts) {
    return this.orInternal(altsOrOpts, 6);
  }
  OR7(altsOrOpts) {
    return this.orInternal(altsOrOpts, 7);
  }
  OR8(altsOrOpts) {
    return this.orInternal(altsOrOpts, 8);
  }
  OR9(altsOrOpts) {
    return this.orInternal(altsOrOpts, 9);
  }
  MANY(actionORMethodDef) {
    this.manyInternal(0, actionORMethodDef);
  }
  MANY1(actionORMethodDef) {
    this.manyInternal(1, actionORMethodDef);
  }
  MANY2(actionORMethodDef) {
    this.manyInternal(2, actionORMethodDef);
  }
  MANY3(actionORMethodDef) {
    this.manyInternal(3, actionORMethodDef);
  }
  MANY4(actionORMethodDef) {
    this.manyInternal(4, actionORMethodDef);
  }
  MANY5(actionORMethodDef) {
    this.manyInternal(5, actionORMethodDef);
  }
  MANY6(actionORMethodDef) {
    this.manyInternal(6, actionORMethodDef);
  }
  MANY7(actionORMethodDef) {
    this.manyInternal(7, actionORMethodDef);
  }
  MANY8(actionORMethodDef) {
    this.manyInternal(8, actionORMethodDef);
  }
  MANY9(actionORMethodDef) {
    this.manyInternal(9, actionORMethodDef);
  }
  MANY_SEP(options) {
    this.manySepFirstInternal(0, options);
  }
  MANY_SEP1(options) {
    this.manySepFirstInternal(1, options);
  }
  MANY_SEP2(options) {
    this.manySepFirstInternal(2, options);
  }
  MANY_SEP3(options) {
    this.manySepFirstInternal(3, options);
  }
  MANY_SEP4(options) {
    this.manySepFirstInternal(4, options);
  }
  MANY_SEP5(options) {
    this.manySepFirstInternal(5, options);
  }
  MANY_SEP6(options) {
    this.manySepFirstInternal(6, options);
  }
  MANY_SEP7(options) {
    this.manySepFirstInternal(7, options);
  }
  MANY_SEP8(options) {
    this.manySepFirstInternal(8, options);
  }
  MANY_SEP9(options) {
    this.manySepFirstInternal(9, options);
  }
  AT_LEAST_ONE(actionORMethodDef) {
    this.atLeastOneInternal(0, actionORMethodDef);
  }
  AT_LEAST_ONE1(actionORMethodDef) {
    return this.atLeastOneInternal(1, actionORMethodDef);
  }
  AT_LEAST_ONE2(actionORMethodDef) {
    this.atLeastOneInternal(2, actionORMethodDef);
  }
  AT_LEAST_ONE3(actionORMethodDef) {
    this.atLeastOneInternal(3, actionORMethodDef);
  }
  AT_LEAST_ONE4(actionORMethodDef) {
    this.atLeastOneInternal(4, actionORMethodDef);
  }
  AT_LEAST_ONE5(actionORMethodDef) {
    this.atLeastOneInternal(5, actionORMethodDef);
  }
  AT_LEAST_ONE6(actionORMethodDef) {
    this.atLeastOneInternal(6, actionORMethodDef);
  }
  AT_LEAST_ONE7(actionORMethodDef) {
    this.atLeastOneInternal(7, actionORMethodDef);
  }
  AT_LEAST_ONE8(actionORMethodDef) {
    this.atLeastOneInternal(8, actionORMethodDef);
  }
  AT_LEAST_ONE9(actionORMethodDef) {
    this.atLeastOneInternal(9, actionORMethodDef);
  }
  AT_LEAST_ONE_SEP(options) {
    this.atLeastOneSepFirstInternal(0, options);
  }
  AT_LEAST_ONE_SEP1(options) {
    this.atLeastOneSepFirstInternal(1, options);
  }
  AT_LEAST_ONE_SEP2(options) {
    this.atLeastOneSepFirstInternal(2, options);
  }
  AT_LEAST_ONE_SEP3(options) {
    this.atLeastOneSepFirstInternal(3, options);
  }
  AT_LEAST_ONE_SEP4(options) {
    this.atLeastOneSepFirstInternal(4, options);
  }
  AT_LEAST_ONE_SEP5(options) {
    this.atLeastOneSepFirstInternal(5, options);
  }
  AT_LEAST_ONE_SEP6(options) {
    this.atLeastOneSepFirstInternal(6, options);
  }
  AT_LEAST_ONE_SEP7(options) {
    this.atLeastOneSepFirstInternal(7, options);
  }
  AT_LEAST_ONE_SEP8(options) {
    this.atLeastOneSepFirstInternal(8, options);
  }
  AT_LEAST_ONE_SEP9(options) {
    this.atLeastOneSepFirstInternal(9, options);
  }
  RULE(name, implementation, config = DEFAULT_RULE_CONFIG) {
    if (includes_default(this.definedRulesNames, name)) {
      const errMsg = defaultGrammarValidatorErrorProvider.buildDuplicateRuleNameError({
        topLevelRule: name,
        grammarName: this.className
      });
      const error = {
        message: errMsg,
        type: ParserDefinitionErrorType.DUPLICATE_RULE_NAME,
        ruleName: name
      };
      this.definitionErrors.push(error);
    }
    this.definedRulesNames.push(name);
    const ruleImplementation = this.defineRule(name, implementation, config);
    this[name] = ruleImplementation;
    return ruleImplementation;
  }
  OVERRIDE_RULE(name, impl, config = DEFAULT_RULE_CONFIG) {
    const ruleErrors = validateRuleIsOverridden(name, this.definedRulesNames, this.className);
    this.definitionErrors = this.definitionErrors.concat(ruleErrors);
    const ruleImplementation = this.defineRule(name, impl, config);
    this[name] = ruleImplementation;
    return ruleImplementation;
  }
  BACKTRACK(grammarRule, args) {
    return function() {
      this.isBackTrackingStack.push(1);
      const orgState = this.saveRecogState();
      try {
        grammarRule.apply(this, args);
        return true;
      } catch (e) {
        if (isRecognitionException(e)) {
          return false;
        } else {
          throw e;
        }
      } finally {
        this.reloadRecogState(orgState);
        this.isBackTrackingStack.pop();
      }
    };
  }
  // GAST export APIs
  getGAstProductions() {
    return this.gastProductionsCache;
  }
  getSerializedGastProductions() {
    return serializeGrammar(values_default(this.gastProductionsCache));
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/recognizer_engine.js
var RecognizerEngine = class {
  initRecognizerEngine(tokenVocabulary, config) {
    this.className = this.constructor.name;
    this.shortRuleNameToFull = {};
    this.fullRuleNameToShort = {};
    this.ruleShortNameIdx = 256;
    this.tokenMatcher = tokenStructuredMatcherNoCategories;
    this.subruleIdx = 0;
    this.definedRulesNames = [];
    this.tokensMap = {};
    this.isBackTrackingStack = [];
    this.RULE_STACK = [];
    this.RULE_OCCURRENCE_STACK = [];
    this.gastProductionsCache = {};
    if (has_default(config, "serializedGrammar")) {
      throw Error("The Parser's configuration can no longer contain a <serializedGrammar> property.\n	See: https://chevrotain.io/docs/changes/BREAKING_CHANGES.html#_6-0-0\n	For Further details.");
    }
    if (isArray_default(tokenVocabulary)) {
      if (isEmpty_default(tokenVocabulary)) {
        throw Error("A Token Vocabulary cannot be empty.\n	Note that the first argument for the parser constructor\n	is no longer a Token vector (since v4.0).");
      }
      if (typeof tokenVocabulary[0].startOffset === "number") {
        throw Error("The Parser constructor no longer accepts a token vector as the first argument.\n	See: https://chevrotain.io/docs/changes/BREAKING_CHANGES.html#_4-0-0\n	For Further details.");
      }
    }
    if (isArray_default(tokenVocabulary)) {
      this.tokensMap = reduce_default(tokenVocabulary, (acc, tokType) => {
        acc[tokType.name] = tokType;
        return acc;
      }, {});
    } else if (has_default(tokenVocabulary, "modes") && every_default(flatten_default(values_default(tokenVocabulary.modes)), isTokenType)) {
      const allTokenTypes2 = flatten_default(values_default(tokenVocabulary.modes));
      const uniqueTokens = uniq_default(allTokenTypes2);
      this.tokensMap = reduce_default(uniqueTokens, (acc, tokType) => {
        acc[tokType.name] = tokType;
        return acc;
      }, {});
    } else if (isObject_default(tokenVocabulary)) {
      this.tokensMap = clone_default(tokenVocabulary);
    } else {
      throw new Error("<tokensDictionary> argument must be An Array of Token constructors, A dictionary of Token constructors or an IMultiModeLexerDefinition");
    }
    this.tokensMap["EOF"] = EOF;
    const allTokenTypes = has_default(tokenVocabulary, "modes") ? flatten_default(values_default(tokenVocabulary.modes)) : values_default(tokenVocabulary);
    const noTokenCategoriesUsed = every_default(allTokenTypes, (tokenConstructor) => isEmpty_default(tokenConstructor.categoryMatches));
    this.tokenMatcher = noTokenCategoriesUsed ? tokenStructuredMatcherNoCategories : tokenStructuredMatcher;
    augmentTokenTypes(values_default(this.tokensMap));
  }
  defineRule(ruleName, impl, config) {
    if (this.selfAnalysisDone) {
      throw Error(`Grammar rule <${ruleName}> may not be defined after the 'performSelfAnalysis' method has been called'
Make sure that all grammar rule definitions are done before 'performSelfAnalysis' is called.`);
    }
    const resyncEnabled = has_default(config, "resyncEnabled") ? config.resyncEnabled : DEFAULT_RULE_CONFIG.resyncEnabled;
    const recoveryValueFunc = has_default(config, "recoveryValueFunc") ? config.recoveryValueFunc : DEFAULT_RULE_CONFIG.recoveryValueFunc;
    const shortName = this.ruleShortNameIdx << BITS_FOR_METHOD_TYPE + BITS_FOR_OCCURRENCE_IDX;
    this.ruleShortNameIdx++;
    this.shortRuleNameToFull[shortName] = ruleName;
    this.fullRuleNameToShort[ruleName] = shortName;
    let invokeRuleWithTry;
    if (this.outputCst === true) {
      invokeRuleWithTry = function invokeRuleWithTry2(...args) {
        try {
          this.ruleInvocationStateUpdate(shortName, ruleName, this.subruleIdx);
          impl.apply(this, args);
          const cst = this.CST_STACK[this.CST_STACK.length - 1];
          this.cstPostRule(cst);
          return cst;
        } catch (e) {
          return this.invokeRuleCatch(e, resyncEnabled, recoveryValueFunc);
        } finally {
          this.ruleFinallyStateUpdate();
        }
      };
    } else {
      invokeRuleWithTry = function invokeRuleWithTryCst(...args) {
        try {
          this.ruleInvocationStateUpdate(shortName, ruleName, this.subruleIdx);
          return impl.apply(this, args);
        } catch (e) {
          return this.invokeRuleCatch(e, resyncEnabled, recoveryValueFunc);
        } finally {
          this.ruleFinallyStateUpdate();
        }
      };
    }
    const wrappedGrammarRule = Object.assign(invokeRuleWithTry, { ruleName, originalGrammarAction: impl });
    return wrappedGrammarRule;
  }
  invokeRuleCatch(e, resyncEnabledConfig, recoveryValueFunc) {
    const isFirstInvokedRule = this.RULE_STACK.length === 1;
    const reSyncEnabled = resyncEnabledConfig && !this.isBackTracking() && this.recoveryEnabled;
    if (isRecognitionException(e)) {
      const recogError = e;
      if (reSyncEnabled) {
        const reSyncTokType = this.findReSyncTokenType();
        if (this.isInCurrentRuleReSyncSet(reSyncTokType)) {
          recogError.resyncedTokens = this.reSyncTo(reSyncTokType);
          if (this.outputCst) {
            const partialCstResult = this.CST_STACK[this.CST_STACK.length - 1];
            partialCstResult.recoveredNode = true;
            return partialCstResult;
          } else {
            return recoveryValueFunc(e);
          }
        } else {
          if (this.outputCst) {
            const partialCstResult = this.CST_STACK[this.CST_STACK.length - 1];
            partialCstResult.recoveredNode = true;
            recogError.partialCstResult = partialCstResult;
          }
          throw recogError;
        }
      } else if (isFirstInvokedRule) {
        this.moveToTerminatedState();
        return recoveryValueFunc(e);
      } else {
        throw recogError;
      }
    } else {
      throw e;
    }
  }
  // Implementation of parsing DSL
  optionInternal(actionORMethodDef, occurrence) {
    const key = this.getKeyForAutomaticLookahead(OPTION_IDX, occurrence);
    return this.optionInternalLogic(actionORMethodDef, occurrence, key);
  }
  optionInternalLogic(actionORMethodDef, occurrence, key) {
    let lookAheadFunc = this.getLaFuncFromCache(key);
    let action;
    if (typeof actionORMethodDef !== "function") {
      action = actionORMethodDef.DEF;
      const predicate = actionORMethodDef.GATE;
      if (predicate !== void 0) {
        const orgLookaheadFunction = lookAheadFunc;
        lookAheadFunc = () => {
          return predicate.call(this) && orgLookaheadFunction.call(this);
        };
      }
    } else {
      action = actionORMethodDef;
    }
    if (lookAheadFunc.call(this) === true) {
      return action.call(this);
    }
    return void 0;
  }
  atLeastOneInternal(prodOccurrence, actionORMethodDef) {
    const laKey = this.getKeyForAutomaticLookahead(AT_LEAST_ONE_IDX, prodOccurrence);
    return this.atLeastOneInternalLogic(prodOccurrence, actionORMethodDef, laKey);
  }
  atLeastOneInternalLogic(prodOccurrence, actionORMethodDef, key) {
    let lookAheadFunc = this.getLaFuncFromCache(key);
    let action;
    if (typeof actionORMethodDef !== "function") {
      action = actionORMethodDef.DEF;
      const predicate = actionORMethodDef.GATE;
      if (predicate !== void 0) {
        const orgLookaheadFunction = lookAheadFunc;
        lookAheadFunc = () => {
          return predicate.call(this) && orgLookaheadFunction.call(this);
        };
      }
    } else {
      action = actionORMethodDef;
    }
    if (lookAheadFunc.call(this) === true) {
      let notStuck = this.doSingleRepetition(action);
      while (lookAheadFunc.call(this) === true && notStuck === true) {
        notStuck = this.doSingleRepetition(action);
      }
    } else {
      throw this.raiseEarlyExitException(prodOccurrence, PROD_TYPE.REPETITION_MANDATORY, actionORMethodDef.ERR_MSG);
    }
    this.attemptInRepetitionRecovery(this.atLeastOneInternal, [prodOccurrence, actionORMethodDef], lookAheadFunc, AT_LEAST_ONE_IDX, prodOccurrence, NextTerminalAfterAtLeastOneWalker);
  }
  atLeastOneSepFirstInternal(prodOccurrence, options) {
    const laKey = this.getKeyForAutomaticLookahead(AT_LEAST_ONE_SEP_IDX, prodOccurrence);
    this.atLeastOneSepFirstInternalLogic(prodOccurrence, options, laKey);
  }
  atLeastOneSepFirstInternalLogic(prodOccurrence, options, key) {
    const action = options.DEF;
    const separator = options.SEP;
    const firstIterationLookaheadFunc = this.getLaFuncFromCache(key);
    if (firstIterationLookaheadFunc.call(this) === true) {
      action.call(this);
      const separatorLookAheadFunc = () => {
        return this.tokenMatcher(this.LA(1), separator);
      };
      while (this.tokenMatcher(this.LA(1), separator) === true) {
        this.CONSUME(separator);
        action.call(this);
      }
      this.attemptInRepetitionRecovery(this.repetitionSepSecondInternal, [
        prodOccurrence,
        separator,
        separatorLookAheadFunc,
        action,
        NextTerminalAfterAtLeastOneSepWalker
      ], separatorLookAheadFunc, AT_LEAST_ONE_SEP_IDX, prodOccurrence, NextTerminalAfterAtLeastOneSepWalker);
    } else {
      throw this.raiseEarlyExitException(prodOccurrence, PROD_TYPE.REPETITION_MANDATORY_WITH_SEPARATOR, options.ERR_MSG);
    }
  }
  manyInternal(prodOccurrence, actionORMethodDef) {
    const laKey = this.getKeyForAutomaticLookahead(MANY_IDX, prodOccurrence);
    return this.manyInternalLogic(prodOccurrence, actionORMethodDef, laKey);
  }
  manyInternalLogic(prodOccurrence, actionORMethodDef, key) {
    let lookaheadFunction = this.getLaFuncFromCache(key);
    let action;
    if (typeof actionORMethodDef !== "function") {
      action = actionORMethodDef.DEF;
      const predicate = actionORMethodDef.GATE;
      if (predicate !== void 0) {
        const orgLookaheadFunction = lookaheadFunction;
        lookaheadFunction = () => {
          return predicate.call(this) && orgLookaheadFunction.call(this);
        };
      }
    } else {
      action = actionORMethodDef;
    }
    let notStuck = true;
    while (lookaheadFunction.call(this) === true && notStuck === true) {
      notStuck = this.doSingleRepetition(action);
    }
    this.attemptInRepetitionRecovery(
      this.manyInternal,
      [prodOccurrence, actionORMethodDef],
      lookaheadFunction,
      MANY_IDX,
      prodOccurrence,
      NextTerminalAfterManyWalker,
      // The notStuck parameter is only relevant when "attemptInRepetitionRecovery"
      // is invoked from manyInternal, in the MANY_SEP case and AT_LEAST_ONE[_SEP]
      // An infinite loop cannot occur as:
      // - Either the lookahead is guaranteed to consume something (Single Token Separator)
      // - AT_LEAST_ONE by definition is guaranteed to consume something (or error out).
      notStuck
    );
  }
  manySepFirstInternal(prodOccurrence, options) {
    const laKey = this.getKeyForAutomaticLookahead(MANY_SEP_IDX, prodOccurrence);
    this.manySepFirstInternalLogic(prodOccurrence, options, laKey);
  }
  manySepFirstInternalLogic(prodOccurrence, options, key) {
    const action = options.DEF;
    const separator = options.SEP;
    const firstIterationLaFunc = this.getLaFuncFromCache(key);
    if (firstIterationLaFunc.call(this) === true) {
      action.call(this);
      const separatorLookAheadFunc = () => {
        return this.tokenMatcher(this.LA(1), separator);
      };
      while (this.tokenMatcher(this.LA(1), separator) === true) {
        this.CONSUME(separator);
        action.call(this);
      }
      this.attemptInRepetitionRecovery(this.repetitionSepSecondInternal, [
        prodOccurrence,
        separator,
        separatorLookAheadFunc,
        action,
        NextTerminalAfterManySepWalker
      ], separatorLookAheadFunc, MANY_SEP_IDX, prodOccurrence, NextTerminalAfterManySepWalker);
    }
  }
  repetitionSepSecondInternal(prodOccurrence, separator, separatorLookAheadFunc, action, nextTerminalAfterWalker) {
    while (separatorLookAheadFunc()) {
      this.CONSUME(separator);
      action.call(this);
    }
    this.attemptInRepetitionRecovery(this.repetitionSepSecondInternal, [
      prodOccurrence,
      separator,
      separatorLookAheadFunc,
      action,
      nextTerminalAfterWalker
    ], separatorLookAheadFunc, AT_LEAST_ONE_SEP_IDX, prodOccurrence, nextTerminalAfterWalker);
  }
  doSingleRepetition(action) {
    const beforeIteration = this.getLexerPosition();
    action.call(this);
    const afterIteration = this.getLexerPosition();
    return afterIteration > beforeIteration;
  }
  orInternal(altsOrOpts, occurrence) {
    const laKey = this.getKeyForAutomaticLookahead(OR_IDX, occurrence);
    const alts = isArray_default(altsOrOpts) ? altsOrOpts : altsOrOpts.DEF;
    const laFunc = this.getLaFuncFromCache(laKey);
    const altIdxToTake = laFunc.call(this, alts);
    if (altIdxToTake !== void 0) {
      const chosenAlternative = alts[altIdxToTake];
      return chosenAlternative.ALT.call(this);
    }
    this.raiseNoAltException(occurrence, altsOrOpts.ERR_MSG);
  }
  ruleFinallyStateUpdate() {
    this.RULE_STACK.pop();
    this.RULE_OCCURRENCE_STACK.pop();
    this.cstFinallyStateUpdate();
    if (this.RULE_STACK.length === 0 && this.isAtEndOfInput() === false) {
      const firstRedundantTok = this.LA(1);
      const errMsg = this.errorMessageProvider.buildNotAllInputParsedMessage({
        firstRedundant: firstRedundantTok,
        ruleName: this.getCurrRuleFullName()
      });
      this.SAVE_ERROR(new NotAllInputParsedException(errMsg, firstRedundantTok));
    }
  }
  subruleInternal(ruleToCall, idx, options) {
    let ruleResult;
    try {
      const args = options !== void 0 ? options.ARGS : void 0;
      this.subruleIdx = idx;
      ruleResult = ruleToCall.apply(this, args);
      this.cstPostNonTerminal(ruleResult, options !== void 0 && options.LABEL !== void 0 ? options.LABEL : ruleToCall.ruleName);
      return ruleResult;
    } catch (e) {
      throw this.subruleInternalError(e, options, ruleToCall.ruleName);
    }
  }
  subruleInternalError(e, options, ruleName) {
    if (isRecognitionException(e) && e.partialCstResult !== void 0) {
      this.cstPostNonTerminal(e.partialCstResult, options !== void 0 && options.LABEL !== void 0 ? options.LABEL : ruleName);
      delete e.partialCstResult;
    }
    throw e;
  }
  consumeInternal(tokType, idx, options) {
    let consumedToken;
    try {
      const nextToken = this.LA(1);
      if (this.tokenMatcher(nextToken, tokType) === true) {
        this.consumeToken();
        consumedToken = nextToken;
      } else {
        this.consumeInternalError(tokType, nextToken, options);
      }
    } catch (eFromConsumption) {
      consumedToken = this.consumeInternalRecovery(tokType, idx, eFromConsumption);
    }
    this.cstPostTerminal(options !== void 0 && options.LABEL !== void 0 ? options.LABEL : tokType.name, consumedToken);
    return consumedToken;
  }
  consumeInternalError(tokType, nextToken, options) {
    let msg;
    const previousToken = this.LA(0);
    if (options !== void 0 && options.ERR_MSG) {
      msg = options.ERR_MSG;
    } else {
      msg = this.errorMessageProvider.buildMismatchTokenMessage({
        expected: tokType,
        actual: nextToken,
        previous: previousToken,
        ruleName: this.getCurrRuleFullName()
      });
    }
    throw this.SAVE_ERROR(new MismatchedTokenException(msg, nextToken, previousToken));
  }
  consumeInternalRecovery(tokType, idx, eFromConsumption) {
    if (this.recoveryEnabled && // TODO: more robust checking of the exception type. Perhaps Typescript extending expressions?
    eFromConsumption.name === "MismatchedTokenException" && !this.isBackTracking()) {
      const follows = this.getFollowsForInRuleRecovery(tokType, idx);
      try {
        return this.tryInRuleRecovery(tokType, follows);
      } catch (eFromInRuleRecovery) {
        if (eFromInRuleRecovery.name === IN_RULE_RECOVERY_EXCEPTION) {
          throw eFromConsumption;
        } else {
          throw eFromInRuleRecovery;
        }
      }
    } else {
      throw eFromConsumption;
    }
  }
  saveRecogState() {
    const savedErrors = this.errors;
    const savedRuleStack = clone_default(this.RULE_STACK);
    return {
      errors: savedErrors,
      lexerState: this.exportLexerState(),
      RULE_STACK: savedRuleStack,
      CST_STACK: this.CST_STACK
    };
  }
  reloadRecogState(newState2) {
    this.errors = newState2.errors;
    this.importLexerState(newState2.lexerState);
    this.RULE_STACK = newState2.RULE_STACK;
  }
  ruleInvocationStateUpdate(shortName, fullName, idxInCallingRule) {
    this.RULE_OCCURRENCE_STACK.push(idxInCallingRule);
    this.RULE_STACK.push(shortName);
    this.cstInvocationStateUpdate(fullName);
  }
  isBackTracking() {
    return this.isBackTrackingStack.length !== 0;
  }
  getCurrRuleFullName() {
    const shortName = this.getLastExplicitRuleShortName();
    return this.shortRuleNameToFull[shortName];
  }
  shortRuleNameToFullName(shortName) {
    return this.shortRuleNameToFull[shortName];
  }
  isAtEndOfInput() {
    return this.tokenMatcher(this.LA(1), EOF);
  }
  reset() {
    this.resetLexerState();
    this.subruleIdx = 0;
    this.isBackTrackingStack = [];
    this.errors = [];
    this.RULE_STACK = [];
    this.CST_STACK = [];
    this.RULE_OCCURRENCE_STACK = [];
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/error_handler.js
var ErrorHandler = class {
  initErrorHandler(config) {
    this._errors = [];
    this.errorMessageProvider = has_default(config, "errorMessageProvider") ? config.errorMessageProvider : DEFAULT_PARSER_CONFIG.errorMessageProvider;
  }
  SAVE_ERROR(error) {
    if (isRecognitionException(error)) {
      error.context = {
        ruleStack: this.getHumanReadableRuleStack(),
        ruleOccurrenceStack: clone_default(this.RULE_OCCURRENCE_STACK)
      };
      this._errors.push(error);
      return error;
    } else {
      throw Error("Trying to save an Error which is not a RecognitionException");
    }
  }
  get errors() {
    return clone_default(this._errors);
  }
  set errors(newErrors) {
    this._errors = newErrors;
  }
  // TODO: consider caching the error message computed information
  raiseEarlyExitException(occurrence, prodType, userDefinedErrMsg) {
    const ruleName = this.getCurrRuleFullName();
    const ruleGrammar = this.getGAstProductions()[ruleName];
    const lookAheadPathsPerAlternative = getLookaheadPathsForOptionalProd(occurrence, ruleGrammar, prodType, this.maxLookahead);
    const insideProdPaths = lookAheadPathsPerAlternative[0];
    const actualTokens = [];
    for (let i = 1; i <= this.maxLookahead; i++) {
      actualTokens.push(this.LA(i));
    }
    const msg = this.errorMessageProvider.buildEarlyExitMessage({
      expectedIterationPaths: insideProdPaths,
      actual: actualTokens,
      previous: this.LA(0),
      customUserDescription: userDefinedErrMsg,
      ruleName
    });
    throw this.SAVE_ERROR(new EarlyExitException(msg, this.LA(1), this.LA(0)));
  }
  // TODO: consider caching the error message computed information
  raiseNoAltException(occurrence, errMsgTypes) {
    const ruleName = this.getCurrRuleFullName();
    const ruleGrammar = this.getGAstProductions()[ruleName];
    const lookAheadPathsPerAlternative = getLookaheadPathsForOr(occurrence, ruleGrammar, this.maxLookahead);
    const actualTokens = [];
    for (let i = 1; i <= this.maxLookahead; i++) {
      actualTokens.push(this.LA(i));
    }
    const previousToken = this.LA(0);
    const errMsg = this.errorMessageProvider.buildNoViableAltMessage({
      expectedPathsPerAlt: lookAheadPathsPerAlternative,
      actual: actualTokens,
      previous: previousToken,
      customUserDescription: errMsgTypes,
      ruleName: this.getCurrRuleFullName()
    });
    throw this.SAVE_ERROR(new NoViableAltException(errMsg, this.LA(1), previousToken));
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/context_assist.js
var ContentAssist = class {
  initContentAssist() {
  }
  computeContentAssist(startRuleName, precedingInput) {
    const startRuleGast = this.gastProductionsCache[startRuleName];
    if (isUndefined_default(startRuleGast)) {
      throw Error(`Rule ->${startRuleName}<- does not exist in this grammar.`);
    }
    return nextPossibleTokensAfter([startRuleGast], precedingInput, this.tokenMatcher, this.maxLookahead);
  }
  // TODO: should this be a member method or a utility? it does not have any state or usage of 'this'...
  // TODO: should this be more explicitly part of the public API?
  getNextPossibleTokenTypes(grammarPath) {
    const topRuleName = head_default(grammarPath.ruleStack);
    const gastProductions = this.getGAstProductions();
    const topProduction = gastProductions[topRuleName];
    const nextPossibleTokenTypes = new NextAfterTokenWalker(topProduction, grammarPath).startWalking();
    return nextPossibleTokenTypes;
  }
};

// node_modules/chevrotain/lib/src/parse/parser/traits/gast_recorder.js
var RECORDING_NULL_OBJECT = {
  description: "This Object indicates the Parser is during Recording Phase"
};
Object.freeze(RECORDING_NULL_OBJECT);
var HANDLE_SEPARATOR = true;
var MAX_METHOD_IDX = Math.pow(2, BITS_FOR_OCCURRENCE_IDX) - 1;
var RFT = createToken({ name: "RECORDING_PHASE_TOKEN", pattern: Lexer.NA });
augmentTokenTypes([RFT]);
var RECORDING_PHASE_TOKEN = createTokenInstance(
  RFT,
  "This IToken indicates the Parser is in Recording Phase\n	See: https://chevrotain.io/docs/guide/internals.html#grammar-recording for details",
  // Using "-1" instead of NaN (as in EOF) because an actual number is less likely to
  // cause errors if the output of LA or CONSUME would be (incorrectly) used during the recording phase.
  -1,
  -1,
  -1,
  -1,
  -1,
  -1
);
Object.freeze(RECORDING_PHASE_TOKEN);
var RECORDING_PHASE_CSTNODE = {
  name: "This CSTNode indicates the Parser is in Recording Phase\n	See: https://chevrotain.io/docs/guide/internals.html#grammar-recording for details",
  children: {}
};
var GastRecorder = class {
  initGastRecorder(config) {
    this.recordingProdStack = [];
    this.RECORDING_PHASE = false;
  }
  enableRecording() {
    this.RECORDING_PHASE = true;
    this.TRACE_INIT("Enable Recording", () => {
      for (let i = 0; i < 10; i++) {
        const idx = i > 0 ? i : "";
        this[`CONSUME${idx}`] = function(arg1, arg2) {
          return this.consumeInternalRecord(arg1, i, arg2);
        };
        this[`SUBRULE${idx}`] = function(arg1, arg2) {
          return this.subruleInternalRecord(arg1, i, arg2);
        };
        this[`OPTION${idx}`] = function(arg1) {
          return this.optionInternalRecord(arg1, i);
        };
        this[`OR${idx}`] = function(arg1) {
          return this.orInternalRecord(arg1, i);
        };
        this[`MANY${idx}`] = function(arg1) {
          this.manyInternalRecord(i, arg1);
        };
        this[`MANY_SEP${idx}`] = function(arg1) {
          this.manySepFirstInternalRecord(i, arg1);
        };
        this[`AT_LEAST_ONE${idx}`] = function(arg1) {
          this.atLeastOneInternalRecord(i, arg1);
        };
        this[`AT_LEAST_ONE_SEP${idx}`] = function(arg1) {
          this.atLeastOneSepFirstInternalRecord(i, arg1);
        };
      }
      this[`consume`] = function(idx, arg1, arg2) {
        return this.consumeInternalRecord(arg1, idx, arg2);
      };
      this[`subrule`] = function(idx, arg1, arg2) {
        return this.subruleInternalRecord(arg1, idx, arg2);
      };
      this[`option`] = function(idx, arg1) {
        return this.optionInternalRecord(arg1, idx);
      };
      this[`or`] = function(idx, arg1) {
        return this.orInternalRecord(arg1, idx);
      };
      this[`many`] = function(idx, arg1) {
        this.manyInternalRecord(idx, arg1);
      };
      this[`atLeastOne`] = function(idx, arg1) {
        this.atLeastOneInternalRecord(idx, arg1);
      };
      this.ACTION = this.ACTION_RECORD;
      this.BACKTRACK = this.BACKTRACK_RECORD;
      this.LA = this.LA_RECORD;
    });
  }
  disableRecording() {
    this.RECORDING_PHASE = false;
    this.TRACE_INIT("Deleting Recording methods", () => {
      const that = this;
      for (let i = 0; i < 10; i++) {
        const idx = i > 0 ? i : "";
        delete that[`CONSUME${idx}`];
        delete that[`SUBRULE${idx}`];
        delete that[`OPTION${idx}`];
        delete that[`OR${idx}`];
        delete that[`MANY${idx}`];
        delete that[`MANY_SEP${idx}`];
        delete that[`AT_LEAST_ONE${idx}`];
        delete that[`AT_LEAST_ONE_SEP${idx}`];
      }
      delete that[`consume`];
      delete that[`subrule`];
      delete that[`option`];
      delete that[`or`];
      delete that[`many`];
      delete that[`atLeastOne`];
      delete that.ACTION;
      delete that.BACKTRACK;
      delete that.LA;
    });
  }
  //   Parser methods are called inside an ACTION?
  //   Maybe try/catch/finally on ACTIONS while disabling the recorders state changes?
  // @ts-expect-error -- noop place holder
  ACTION_RECORD(impl) {
  }
  // Executing backtracking logic will break our recording logic assumptions
  BACKTRACK_RECORD(grammarRule, args) {
    return () => true;
  }
  // LA is part of the official API and may be used for custom lookahead logic
  // by end users who may forget to wrap it in ACTION or inside a GATE
  LA_RECORD(howMuch) {
    return END_OF_FILE;
  }
  topLevelRuleRecord(name, def) {
    try {
      const newTopLevelRule = new Rule({ definition: [], name });
      newTopLevelRule.name = name;
      this.recordingProdStack.push(newTopLevelRule);
      def.call(this);
      this.recordingProdStack.pop();
      return newTopLevelRule;
    } catch (originalError) {
      if (originalError.KNOWN_RECORDER_ERROR !== true) {
        try {
          originalError.message = originalError.message + '\n	 This error was thrown during the "grammar recording phase" For more info see:\n	https://chevrotain.io/docs/guide/internals.html#grammar-recording';
        } catch (mutabilityError) {
          throw originalError;
        }
      }
      throw originalError;
    }
  }
  // Implementation of parsing DSL
  optionInternalRecord(actionORMethodDef, occurrence) {
    return recordProd.call(this, Option, actionORMethodDef, occurrence);
  }
  atLeastOneInternalRecord(occurrence, actionORMethodDef) {
    recordProd.call(this, RepetitionMandatory, actionORMethodDef, occurrence);
  }
  atLeastOneSepFirstInternalRecord(occurrence, options) {
    recordProd.call(this, RepetitionMandatoryWithSeparator, options, occurrence, HANDLE_SEPARATOR);
  }
  manyInternalRecord(occurrence, actionORMethodDef) {
    recordProd.call(this, Repetition, actionORMethodDef, occurrence);
  }
  manySepFirstInternalRecord(occurrence, options) {
    recordProd.call(this, RepetitionWithSeparator, options, occurrence, HANDLE_SEPARATOR);
  }
  orInternalRecord(altsOrOpts, occurrence) {
    return recordOrProd.call(this, altsOrOpts, occurrence);
  }
  subruleInternalRecord(ruleToCall, occurrence, options) {
    assertMethodIdxIsValid(occurrence);
    if (!ruleToCall || has_default(ruleToCall, "ruleName") === false) {
      const error = new Error(`<SUBRULE${getIdxSuffix(occurrence)}> argument is invalid expecting a Parser method reference but got: <${JSON.stringify(ruleToCall)}>
 inside top level rule: <${this.recordingProdStack[0].name}>`);
      error.KNOWN_RECORDER_ERROR = true;
      throw error;
    }
    const prevProd = last_default(this.recordingProdStack);
    const ruleName = ruleToCall.ruleName;
    const newNoneTerminal = new NonTerminal({
      idx: occurrence,
      nonTerminalName: ruleName,
      label: options === null || options === void 0 ? void 0 : options.LABEL,
      // The resolving of the `referencedRule` property will be done once all the Rule's GASTs have been created
      referencedRule: void 0
    });
    prevProd.definition.push(newNoneTerminal);
    return this.outputCst ? RECORDING_PHASE_CSTNODE : RECORDING_NULL_OBJECT;
  }
  consumeInternalRecord(tokType, occurrence, options) {
    assertMethodIdxIsValid(occurrence);
    if (!hasShortKeyProperty(tokType)) {
      const error = new Error(`<CONSUME${getIdxSuffix(occurrence)}> argument is invalid expecting a TokenType reference but got: <${JSON.stringify(tokType)}>
 inside top level rule: <${this.recordingProdStack[0].name}>`);
      error.KNOWN_RECORDER_ERROR = true;
      throw error;
    }
    const prevProd = last_default(this.recordingProdStack);
    const newNoneTerminal = new Terminal({
      idx: occurrence,
      terminalType: tokType,
      label: options === null || options === void 0 ? void 0 : options.LABEL
    });
    prevProd.definition.push(newNoneTerminal);
    return RECORDING_PHASE_TOKEN;
  }
};
function recordProd(prodConstructor, mainProdArg, occurrence, handleSep = false) {
  assertMethodIdxIsValid(occurrence);
  const prevProd = last_default(this.recordingProdStack);
  const grammarAction = isFunction_default(mainProdArg) ? mainProdArg : mainProdArg.DEF;
  const newProd = new prodConstructor({ definition: [], idx: occurrence });
  if (handleSep) {
    newProd.separator = mainProdArg.SEP;
  }
  if (has_default(mainProdArg, "MAX_LOOKAHEAD")) {
    newProd.maxLookahead = mainProdArg.MAX_LOOKAHEAD;
  }
  this.recordingProdStack.push(newProd);
  grammarAction.call(this);
  prevProd.definition.push(newProd);
  this.recordingProdStack.pop();
  return RECORDING_NULL_OBJECT;
}
function recordOrProd(mainProdArg, occurrence) {
  assertMethodIdxIsValid(occurrence);
  const prevProd = last_default(this.recordingProdStack);
  const hasOptions = isArray_default(mainProdArg) === false;
  const alts = hasOptions === false ? mainProdArg : mainProdArg.DEF;
  const newOrProd = new Alternation({
    definition: [],
    idx: occurrence,
    ignoreAmbiguities: hasOptions && mainProdArg.IGNORE_AMBIGUITIES === true
  });
  if (has_default(mainProdArg, "MAX_LOOKAHEAD")) {
    newOrProd.maxLookahead = mainProdArg.MAX_LOOKAHEAD;
  }
  const hasPredicates = some_default(alts, (currAlt) => isFunction_default(currAlt.GATE));
  newOrProd.hasPredicates = hasPredicates;
  prevProd.definition.push(newOrProd);
  forEach_default(alts, (currAlt) => {
    const currAltFlat = new Alternative({ definition: [] });
    newOrProd.definition.push(currAltFlat);
    if (has_default(currAlt, "IGNORE_AMBIGUITIES")) {
      currAltFlat.ignoreAmbiguities = currAlt.IGNORE_AMBIGUITIES;
    } else if (has_default(currAlt, "GATE")) {
      currAltFlat.ignoreAmbiguities = true;
    }
    this.recordingProdStack.push(currAltFlat);
    currAlt.ALT.call(this);
    this.recordingProdStack.pop();
  });
  return RECORDING_NULL_OBJECT;
}
function getIdxSuffix(idx) {
  return idx === 0 ? "" : `${idx}`;
}
function assertMethodIdxIsValid(idx) {
  if (idx < 0 || idx > MAX_METHOD_IDX) {
    const error = new Error(
      // The stack trace will contain all the needed details
      `Invalid DSL Method idx value: <${idx}>
	Idx value must be a none negative value smaller than ${MAX_METHOD_IDX + 1}`
    );
    error.KNOWN_RECORDER_ERROR = true;
    throw error;
  }
}

// node_modules/chevrotain/lib/src/parse/parser/traits/perf_tracer.js
var PerformanceTracer = class {
  initPerformanceTracer(config) {
    if (has_default(config, "traceInitPerf")) {
      const userTraceInitPerf = config.traceInitPerf;
      const traceIsNumber = typeof userTraceInitPerf === "number";
      this.traceInitMaxIdent = traceIsNumber ? userTraceInitPerf : Infinity;
      this.traceInitPerf = traceIsNumber ? userTraceInitPerf > 0 : userTraceInitPerf;
    } else {
      this.traceInitMaxIdent = 0;
      this.traceInitPerf = DEFAULT_PARSER_CONFIG.traceInitPerf;
    }
    this.traceInitIndent = -1;
  }
  TRACE_INIT(phaseDesc, phaseImpl) {
    if (this.traceInitPerf === true) {
      this.traceInitIndent++;
      const indent = new Array(this.traceInitIndent + 1).join("	");
      if (this.traceInitIndent < this.traceInitMaxIdent) {
        console.log(`${indent}--> <${phaseDesc}>`);
      }
      const { time, value } = timer(phaseImpl);
      const traceMethod = time > 10 ? console.warn : console.log;
      if (this.traceInitIndent < this.traceInitMaxIdent) {
        traceMethod(`${indent}<-- <${phaseDesc}> time: ${time}ms`);
      }
      this.traceInitIndent--;
      return value;
    } else {
      return phaseImpl();
    }
  }
};

// node_modules/chevrotain/lib/src/parse/parser/utils/apply_mixins.js
function applyMixins(derivedCtor, baseCtors) {
  baseCtors.forEach((baseCtor) => {
    const baseProto = baseCtor.prototype;
    Object.getOwnPropertyNames(baseProto).forEach((propName) => {
      if (propName === "constructor") {
        return;
      }
      const basePropDescriptor = Object.getOwnPropertyDescriptor(baseProto, propName);
      if (basePropDescriptor && (basePropDescriptor.get || basePropDescriptor.set)) {
        Object.defineProperty(derivedCtor.prototype, propName, basePropDescriptor);
      } else {
        derivedCtor.prototype[propName] = baseCtor.prototype[propName];
      }
    });
  });
}

// node_modules/chevrotain/lib/src/parse/parser/parser.js
var END_OF_FILE = createTokenInstance(EOF, "", NaN, NaN, NaN, NaN, NaN, NaN);
Object.freeze(END_OF_FILE);
var DEFAULT_PARSER_CONFIG = Object.freeze({
  recoveryEnabled: false,
  maxLookahead: 3,
  dynamicTokensEnabled: false,
  outputCst: true,
  errorMessageProvider: defaultParserErrorProvider,
  nodeLocationTracking: "none",
  traceInitPerf: false,
  skipValidations: false
});
var DEFAULT_RULE_CONFIG = Object.freeze({
  recoveryValueFunc: () => void 0,
  resyncEnabled: true
});
var ParserDefinitionErrorType;
(function(ParserDefinitionErrorType2) {
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["INVALID_RULE_NAME"] = 0] = "INVALID_RULE_NAME";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["DUPLICATE_RULE_NAME"] = 1] = "DUPLICATE_RULE_NAME";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["INVALID_RULE_OVERRIDE"] = 2] = "INVALID_RULE_OVERRIDE";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["DUPLICATE_PRODUCTIONS"] = 3] = "DUPLICATE_PRODUCTIONS";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["UNRESOLVED_SUBRULE_REF"] = 4] = "UNRESOLVED_SUBRULE_REF";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["LEFT_RECURSION"] = 5] = "LEFT_RECURSION";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["NONE_LAST_EMPTY_ALT"] = 6] = "NONE_LAST_EMPTY_ALT";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["AMBIGUOUS_ALTS"] = 7] = "AMBIGUOUS_ALTS";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["CONFLICT_TOKENS_RULES_NAMESPACE"] = 8] = "CONFLICT_TOKENS_RULES_NAMESPACE";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["INVALID_TOKEN_NAME"] = 9] = "INVALID_TOKEN_NAME";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["NO_NON_EMPTY_LOOKAHEAD"] = 10] = "NO_NON_EMPTY_LOOKAHEAD";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["AMBIGUOUS_PREFIX_ALTS"] = 11] = "AMBIGUOUS_PREFIX_ALTS";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["TOO_MANY_ALTS"] = 12] = "TOO_MANY_ALTS";
  ParserDefinitionErrorType2[ParserDefinitionErrorType2["CUSTOM_LOOKAHEAD_VALIDATION"] = 13] = "CUSTOM_LOOKAHEAD_VALIDATION";
})(ParserDefinitionErrorType || (ParserDefinitionErrorType = {}));
function EMPTY_ALT(value = void 0) {
  return function() {
    return value;
  };
}
var Parser = class _Parser {
  /**
   *  @deprecated use the **instance** method with the same name instead
   */
  static performSelfAnalysis(parserInstance) {
    throw Error("The **static** `performSelfAnalysis` method has been deprecated.	\nUse the **instance** method with the same name instead.");
  }
  performSelfAnalysis() {
    this.TRACE_INIT("performSelfAnalysis", () => {
      let defErrorsMsgs;
      this.selfAnalysisDone = true;
      const className = this.className;
      this.TRACE_INIT("toFastProps", () => {
        toFastProperties(this);
      });
      this.TRACE_INIT("Grammar Recording", () => {
        try {
          this.enableRecording();
          forEach_default(this.definedRulesNames, (currRuleName) => {
            const wrappedRule = this[currRuleName];
            const originalGrammarAction = wrappedRule["originalGrammarAction"];
            let recordedRuleGast;
            this.TRACE_INIT(`${currRuleName} Rule`, () => {
              recordedRuleGast = this.topLevelRuleRecord(currRuleName, originalGrammarAction);
            });
            this.gastProductionsCache[currRuleName] = recordedRuleGast;
          });
        } finally {
          this.disableRecording();
        }
      });
      let resolverErrors = [];
      this.TRACE_INIT("Grammar Resolving", () => {
        resolverErrors = resolveGrammar2({
          rules: values_default(this.gastProductionsCache)
        });
        this.definitionErrors = this.definitionErrors.concat(resolverErrors);
      });
      this.TRACE_INIT("Grammar Validations", () => {
        if (isEmpty_default(resolverErrors) && this.skipValidations === false) {
          const validationErrors = validateGrammar2({
            rules: values_default(this.gastProductionsCache),
            tokenTypes: values_default(this.tokensMap),
            errMsgProvider: defaultGrammarValidatorErrorProvider,
            grammarName: className
          });
          const lookaheadValidationErrors = validateLookahead({
            lookaheadStrategy: this.lookaheadStrategy,
            rules: values_default(this.gastProductionsCache),
            tokenTypes: values_default(this.tokensMap),
            grammarName: className
          });
          this.definitionErrors = this.definitionErrors.concat(validationErrors, lookaheadValidationErrors);
        }
      });
      if (isEmpty_default(this.definitionErrors)) {
        if (this.recoveryEnabled) {
          this.TRACE_INIT("computeAllProdsFollows", () => {
            const allFollows = computeAllProdsFollows(values_default(this.gastProductionsCache));
            this.resyncFollows = allFollows;
          });
        }
        this.TRACE_INIT("ComputeLookaheadFunctions", () => {
          var _a, _b;
          (_b = (_a = this.lookaheadStrategy).initialize) === null || _b === void 0 ? void 0 : _b.call(_a, {
            rules: values_default(this.gastProductionsCache)
          });
          this.preComputeLookaheadFunctions(values_default(this.gastProductionsCache));
        });
      }
      if (!_Parser.DEFER_DEFINITION_ERRORS_HANDLING && !isEmpty_default(this.definitionErrors)) {
        defErrorsMsgs = map_default(this.definitionErrors, (defError) => defError.message);
        throw new Error(`Parser Definition Errors detected:
 ${defErrorsMsgs.join("\n-------------------------------\n")}`);
      }
    });
  }
  constructor(tokenVocabulary, config) {
    this.definitionErrors = [];
    this.selfAnalysisDone = false;
    const that = this;
    that.initErrorHandler(config);
    that.initLexerAdapter();
    that.initLooksAhead(config);
    that.initRecognizerEngine(tokenVocabulary, config);
    that.initRecoverable(config);
    that.initTreeBuilder(config);
    that.initContentAssist();
    that.initGastRecorder(config);
    that.initPerformanceTracer(config);
    if (has_default(config, "ignoredIssues")) {
      throw new Error("The <ignoredIssues> IParserConfig property has been deprecated.\n	Please use the <IGNORE_AMBIGUITIES> flag on the relevant DSL method instead.\n	See: https://chevrotain.io/docs/guide/resolving_grammar_errors.html#IGNORING_AMBIGUITIES\n	For further details.");
    }
    this.skipValidations = has_default(config, "skipValidations") ? config.skipValidations : DEFAULT_PARSER_CONFIG.skipValidations;
  }
};
Parser.DEFER_DEFINITION_ERRORS_HANDLING = false;
applyMixins(Parser, [
  Recoverable,
  LooksAhead,
  TreeBuilder,
  LexerAdapter,
  RecognizerEngine,
  RecognizerApi,
  ErrorHandler,
  ContentAssist,
  GastRecorder,
  PerformanceTracer
]);
var EmbeddedActionsParser = class extends Parser {
  constructor(tokenVocabulary, config = DEFAULT_PARSER_CONFIG) {
    const configClone = clone_default(config);
    configClone.outputCst = false;
    super(tokenVocabulary, configClone);
  }
};

// node_modules/chevrotain-allstar/lib/atn.js
function buildATNKey(rule, type, occurrence) {
  return `${rule.name}_${type}_${occurrence}`;
}
var ATN_BASIC = 1;
var ATN_RULE_START = 2;
var ATN_PLUS_BLOCK_START = 4;
var ATN_STAR_BLOCK_START = 5;
var ATN_RULE_STOP = 7;
var ATN_BLOCK_END = 8;
var ATN_STAR_LOOP_BACK = 9;
var ATN_STAR_LOOP_ENTRY = 10;
var ATN_PLUS_LOOP_BACK = 11;
var ATN_LOOP_END = 12;
var AbstractTransition = class {
  constructor(target) {
    this.target = target;
  }
  isEpsilon() {
    return false;
  }
};
var AtomTransition = class extends AbstractTransition {
  constructor(target, tokenType) {
    super(target);
    this.tokenType = tokenType;
  }
};
var EpsilonTransition = class extends AbstractTransition {
  constructor(target) {
    super(target);
  }
  isEpsilon() {
    return true;
  }
};
var RuleTransition = class extends AbstractTransition {
  constructor(ruleStart, rule, followState) {
    super(ruleStart);
    this.rule = rule;
    this.followState = followState;
  }
  isEpsilon() {
    return true;
  }
};
function createATN(rules) {
  const atn = {
    decisionMap: {},
    decisionStates: [],
    ruleToStartState: /* @__PURE__ */ new Map(),
    ruleToStopState: /* @__PURE__ */ new Map(),
    states: []
  };
  createRuleStartAndStopATNStates(atn, rules);
  const ruleLength = rules.length;
  for (let i = 0; i < ruleLength; i++) {
    const rule = rules[i];
    const ruleBlock = block(atn, rule, rule);
    if (ruleBlock === void 0) {
      continue;
    }
    buildRuleHandle(atn, rule, ruleBlock);
  }
  return atn;
}
function createRuleStartAndStopATNStates(atn, rules) {
  const ruleLength = rules.length;
  for (let i = 0; i < ruleLength; i++) {
    const rule = rules[i];
    const start = newState(atn, rule, void 0, {
      type: ATN_RULE_START
    });
    const stop = newState(atn, rule, void 0, {
      type: ATN_RULE_STOP
    });
    start.stop = stop;
    atn.ruleToStartState.set(rule, start);
    atn.ruleToStopState.set(rule, stop);
  }
}
function atom(atn, rule, production) {
  if (production instanceof Terminal) {
    return tokenRef(atn, rule, production.terminalType, production);
  } else if (production instanceof NonTerminal) {
    return ruleRef(atn, rule, production);
  } else if (production instanceof Alternation) {
    return alternation(atn, rule, production);
  } else if (production instanceof Option) {
    return option(atn, rule, production);
  } else if (production instanceof Repetition) {
    return repetition(atn, rule, production);
  } else if (production instanceof RepetitionWithSeparator) {
    return repetitionSep(atn, rule, production);
  } else if (production instanceof RepetitionMandatory) {
    return repetitionMandatory(atn, rule, production);
  } else if (production instanceof RepetitionMandatoryWithSeparator) {
    return repetitionMandatorySep(atn, rule, production);
  } else {
    return block(atn, rule, production);
  }
}
function repetition(atn, rule, repetition2) {
  const starState = newState(atn, rule, repetition2, {
    type: ATN_STAR_BLOCK_START
  });
  defineDecisionState(atn, starState);
  const handle = makeAlts(atn, rule, starState, repetition2, block(atn, rule, repetition2));
  return star(atn, rule, repetition2, handle);
}
function repetitionSep(atn, rule, repetition2) {
  const starState = newState(atn, rule, repetition2, {
    type: ATN_STAR_BLOCK_START
  });
  defineDecisionState(atn, starState);
  const handle = makeAlts(atn, rule, starState, repetition2, block(atn, rule, repetition2));
  const sep = tokenRef(atn, rule, repetition2.separator, repetition2);
  return star(atn, rule, repetition2, handle, sep);
}
function repetitionMandatory(atn, rule, repetition2) {
  const plusState = newState(atn, rule, repetition2, {
    type: ATN_PLUS_BLOCK_START
  });
  defineDecisionState(atn, plusState);
  const handle = makeAlts(atn, rule, plusState, repetition2, block(atn, rule, repetition2));
  return plus(atn, rule, repetition2, handle);
}
function repetitionMandatorySep(atn, rule, repetition2) {
  const plusState = newState(atn, rule, repetition2, {
    type: ATN_PLUS_BLOCK_START
  });
  defineDecisionState(atn, plusState);
  const handle = makeAlts(atn, rule, plusState, repetition2, block(atn, rule, repetition2));
  const sep = tokenRef(atn, rule, repetition2.separator, repetition2);
  return plus(atn, rule, repetition2, handle, sep);
}
function alternation(atn, rule, alternation2) {
  const start = newState(atn, rule, alternation2, {
    type: ATN_BASIC
  });
  defineDecisionState(atn, start);
  const alts = map_default(alternation2.definition, (e) => atom(atn, rule, e));
  const handle = makeAlts(atn, rule, start, alternation2, ...alts);
  return handle;
}
function option(atn, rule, option2) {
  const start = newState(atn, rule, option2, {
    type: ATN_BASIC
  });
  defineDecisionState(atn, start);
  const handle = makeAlts(atn, rule, start, option2, block(atn, rule, option2));
  return optional(atn, rule, option2, handle);
}
function block(atn, rule, block2) {
  const handles = filter_default(map_default(block2.definition, (e) => atom(atn, rule, e)), (e) => e !== void 0);
  if (handles.length === 1) {
    return handles[0];
  } else if (handles.length === 0) {
    return void 0;
  } else {
    return makeBlock(atn, handles);
  }
}
function plus(atn, rule, plus2, handle, sep) {
  const blkStart = handle.left;
  const blkEnd = handle.right;
  const loop = newState(atn, rule, plus2, {
    type: ATN_PLUS_LOOP_BACK
  });
  defineDecisionState(atn, loop);
  const end = newState(atn, rule, plus2, {
    type: ATN_LOOP_END
  });
  blkStart.loopback = loop;
  end.loopback = loop;
  atn.decisionMap[buildATNKey(rule, sep ? "RepetitionMandatoryWithSeparator" : "RepetitionMandatory", plus2.idx)] = loop;
  epsilon(blkEnd, loop);
  if (sep === void 0) {
    epsilon(loop, blkStart);
    epsilon(loop, end);
  } else {
    epsilon(loop, end);
    epsilon(loop, sep.left);
    epsilon(sep.right, blkStart);
  }
  return {
    left: blkStart,
    right: end
  };
}
function star(atn, rule, star2, handle, sep) {
  const start = handle.left;
  const end = handle.right;
  const entry = newState(atn, rule, star2, {
    type: ATN_STAR_LOOP_ENTRY
  });
  defineDecisionState(atn, entry);
  const loopEnd = newState(atn, rule, star2, {
    type: ATN_LOOP_END
  });
  const loop = newState(atn, rule, star2, {
    type: ATN_STAR_LOOP_BACK
  });
  entry.loopback = loop;
  loopEnd.loopback = loop;
  epsilon(entry, start);
  epsilon(entry, loopEnd);
  epsilon(end, loop);
  if (sep !== void 0) {
    epsilon(loop, loopEnd);
    epsilon(loop, sep.left);
    epsilon(sep.right, start);
  } else {
    epsilon(loop, entry);
  }
  atn.decisionMap[buildATNKey(rule, sep ? "RepetitionWithSeparator" : "Repetition", star2.idx)] = entry;
  return {
    left: entry,
    right: loopEnd
  };
}
function optional(atn, rule, optional2, handle) {
  const start = handle.left;
  const end = handle.right;
  epsilon(start, end);
  atn.decisionMap[buildATNKey(rule, "Option", optional2.idx)] = start;
  return handle;
}
function defineDecisionState(atn, state) {
  atn.decisionStates.push(state);
  state.decision = atn.decisionStates.length - 1;
  return state.decision;
}
function makeAlts(atn, rule, start, production, ...alts) {
  const end = newState(atn, rule, production, {
    type: ATN_BLOCK_END,
    start
  });
  start.end = end;
  for (const alt of alts) {
    if (alt !== void 0) {
      epsilon(start, alt.left);
      epsilon(alt.right, end);
    } else {
      epsilon(start, end);
    }
  }
  const handle = {
    left: start,
    right: end
  };
  atn.decisionMap[buildATNKey(rule, getProdType2(production), production.idx)] = start;
  return handle;
}
function getProdType2(production) {
  if (production instanceof Alternation) {
    return "Alternation";
  } else if (production instanceof Option) {
    return "Option";
  } else if (production instanceof Repetition) {
    return "Repetition";
  } else if (production instanceof RepetitionWithSeparator) {
    return "RepetitionWithSeparator";
  } else if (production instanceof RepetitionMandatory) {
    return "RepetitionMandatory";
  } else if (production instanceof RepetitionMandatoryWithSeparator) {
    return "RepetitionMandatoryWithSeparator";
  } else {
    throw new Error("Invalid production type encountered");
  }
}
function makeBlock(atn, alts) {
  const altsLength = alts.length;
  for (let i = 0; i < altsLength - 1; i++) {
    const handle = alts[i];
    let transition;
    if (handle.left.transitions.length === 1) {
      transition = handle.left.transitions[0];
    }
    const isRuleTransition = transition instanceof RuleTransition;
    const ruleTransition = transition;
    const next = alts[i + 1].left;
    if (handle.left.type === ATN_BASIC && handle.right.type === ATN_BASIC && transition !== void 0 && (isRuleTransition && ruleTransition.followState === handle.right || transition.target === handle.right)) {
      if (isRuleTransition) {
        ruleTransition.followState = next;
      } else {
        transition.target = next;
      }
      removeState(atn, handle.right);
    } else {
      epsilon(handle.right, next);
    }
  }
  const first2 = alts[0];
  const last2 = alts[altsLength - 1];
  return {
    left: first2.left,
    right: last2.right
  };
}
function tokenRef(atn, rule, tokenType, production) {
  const left = newState(atn, rule, production, {
    type: ATN_BASIC
  });
  const right = newState(atn, rule, production, {
    type: ATN_BASIC
  });
  addTransition(left, new AtomTransition(right, tokenType));
  return {
    left,
    right
  };
}
function ruleRef(atn, currentRule, nonTerminal) {
  const rule = nonTerminal.referencedRule;
  const start = atn.ruleToStartState.get(rule);
  const left = newState(atn, currentRule, nonTerminal, {
    type: ATN_BASIC
  });
  const right = newState(atn, currentRule, nonTerminal, {
    type: ATN_BASIC
  });
  const call = new RuleTransition(start, rule, right);
  addTransition(left, call);
  return {
    left,
    right
  };
}
function buildRuleHandle(atn, rule, block2) {
  const start = atn.ruleToStartState.get(rule);
  epsilon(start, block2.left);
  const stop = atn.ruleToStopState.get(rule);
  epsilon(block2.right, stop);
  const handle = {
    left: start,
    right: stop
  };
  return handle;
}
function epsilon(a, b) {
  const transition = new EpsilonTransition(b);
  addTransition(a, transition);
}
function newState(atn, rule, production, partial2) {
  const t = Object.assign({
    atn,
    production,
    epsilonOnlyTransitions: false,
    rule,
    transitions: [],
    nextTokenWithinRule: [],
    stateNumber: atn.states.length
  }, partial2);
  atn.states.push(t);
  return t;
}
function addTransition(state, transition) {
  if (state.transitions.length === 0) {
    state.epsilonOnlyTransitions = transition.isEpsilon();
  }
  state.transitions.push(transition);
}
function removeState(atn, state) {
  atn.states.splice(atn.states.indexOf(state), 1);
}

// node_modules/chevrotain-allstar/lib/dfa.js
var DFA_ERROR = {};
var ATNConfigSet = class {
  constructor() {
    this.map = {};
    this.configs = [];
  }
  get size() {
    return this.configs.length;
  }
  finalize() {
    this.map = {};
  }
  add(config) {
    const key = getATNConfigKey(config);
    if (!(key in this.map)) {
      this.map[key] = this.configs.length;
      this.configs.push(config);
    }
  }
  get elements() {
    return this.configs;
  }
  get alts() {
    return map_default(this.configs, (e) => e.alt);
  }
  get key() {
    let value = "";
    for (const k in this.map) {
      value += k + ":";
    }
    return value;
  }
};
function getATNConfigKey(config, alt = true) {
  return `${alt ? `a${config.alt}` : ""}s${config.state.stateNumber}:${config.stack.map((e) => e.stateNumber.toString()).join("_")}`;
}

// node_modules/chevrotain-allstar/lib/all-star-lookahead.js
function createDFACache(startState, decision) {
  const map2 = {};
  return (predicateSet) => {
    const key = predicateSet.toString();
    let existing = map2[key];
    if (existing !== void 0) {
      return existing;
    } else {
      existing = {
        atnStartState: startState,
        decision,
        states: {}
      };
      map2[key] = existing;
      return existing;
    }
  };
}
var PredicateSet = class {
  constructor() {
    this.predicates = [];
  }
  is(index) {
    return index >= this.predicates.length || this.predicates[index];
  }
  set(index, value) {
    this.predicates[index] = value;
  }
  toString() {
    let value = "";
    const size2 = this.predicates.length;
    for (let i = 0; i < size2; i++) {
      value += this.predicates[i] === true ? "1" : "0";
    }
    return value;
  }
};
var EMPTY_PREDICATES = new PredicateSet();
var LLStarLookaheadStrategy = class extends LLkLookaheadStrategy {
  constructor(options) {
    var _a;
    super();
    this.logging = (_a = options === null || options === void 0 ? void 0 : options.logging) !== null && _a !== void 0 ? _a : (message) => console.log(message);
  }
  initialize(options) {
    this.atn = createATN(options.rules);
    this.dfas = initATNSimulator(this.atn);
  }
  validateAmbiguousAlternationAlternatives() {
    return [];
  }
  validateEmptyOrAlternatives() {
    return [];
  }
  buildLookaheadForAlternation(options) {
    const { prodOccurrence, rule, hasPredicates, dynamicTokensEnabled } = options;
    const dfas = this.dfas;
    const logging = this.logging;
    const key = buildATNKey(rule, "Alternation", prodOccurrence);
    const decisionState = this.atn.decisionMap[key];
    const decisionIndex = decisionState.decision;
    const partialAlts = map_default(getLookaheadPaths({
      maxLookahead: 1,
      occurrence: prodOccurrence,
      prodType: "Alternation",
      rule
    }), (currAlt) => map_default(currAlt, (path) => path[0]));
    if (isLL1Sequence(partialAlts, false) && !dynamicTokensEnabled) {
      const choiceToAlt = reduce_default(partialAlts, (result2, currAlt, idx) => {
        forEach_default(currAlt, (currTokType) => {
          if (currTokType) {
            result2[currTokType.tokenTypeIdx] = idx;
            forEach_default(currTokType.categoryMatches, (currExtendingType) => {
              result2[currExtendingType] = idx;
            });
          }
        });
        return result2;
      }, {});
      if (hasPredicates) {
        return function(orAlts) {
          var _a;
          const nextToken = this.LA(1);
          const prediction = choiceToAlt[nextToken.tokenTypeIdx];
          if (orAlts !== void 0 && prediction !== void 0) {
            const gate = (_a = orAlts[prediction]) === null || _a === void 0 ? void 0 : _a.GATE;
            if (gate !== void 0 && gate.call(this) === false) {
              return void 0;
            }
          }
          return prediction;
        };
      } else {
        return function() {
          const nextToken = this.LA(1);
          return choiceToAlt[nextToken.tokenTypeIdx];
        };
      }
    } else if (hasPredicates) {
      return function(orAlts) {
        const predicates = new PredicateSet();
        const length = orAlts === void 0 ? 0 : orAlts.length;
        for (let i = 0; i < length; i++) {
          const gate = orAlts === null || orAlts === void 0 ? void 0 : orAlts[i].GATE;
          predicates.set(i, gate === void 0 || gate.call(this));
        }
        const result2 = adaptivePredict.call(this, dfas, decisionIndex, predicates, logging);
        return typeof result2 === "number" ? result2 : void 0;
      };
    } else {
      return function() {
        const result2 = adaptivePredict.call(this, dfas, decisionIndex, EMPTY_PREDICATES, logging);
        return typeof result2 === "number" ? result2 : void 0;
      };
    }
  }
  buildLookaheadForOptional(options) {
    const { prodOccurrence, rule, prodType, dynamicTokensEnabled } = options;
    const dfas = this.dfas;
    const logging = this.logging;
    const key = buildATNKey(rule, prodType, prodOccurrence);
    const decisionState = this.atn.decisionMap[key];
    const decisionIndex = decisionState.decision;
    const alts = map_default(getLookaheadPaths({
      maxLookahead: 1,
      occurrence: prodOccurrence,
      prodType,
      rule
    }), (e) => {
      return map_default(e, (g) => g[0]);
    });
    if (isLL1Sequence(alts) && alts[0][0] && !dynamicTokensEnabled) {
      const alt = alts[0];
      const singleTokensTypes = flatten_default(alt);
      if (singleTokensTypes.length === 1 && isEmpty_default(singleTokensTypes[0].categoryMatches)) {
        const expectedTokenType = singleTokensTypes[0];
        const expectedTokenUniqueKey = expectedTokenType.tokenTypeIdx;
        return function() {
          return this.LA(1).tokenTypeIdx === expectedTokenUniqueKey;
        };
      } else {
        const choiceToAlt = reduce_default(singleTokensTypes, (result2, currTokType) => {
          if (currTokType !== void 0) {
            result2[currTokType.tokenTypeIdx] = true;
            forEach_default(currTokType.categoryMatches, (currExtendingType) => {
              result2[currExtendingType] = true;
            });
          }
          return result2;
        }, {});
        return function() {
          const nextToken = this.LA(1);
          return choiceToAlt[nextToken.tokenTypeIdx] === true;
        };
      }
    }
    return function() {
      const result2 = adaptivePredict.call(this, dfas, decisionIndex, EMPTY_PREDICATES, logging);
      return typeof result2 === "object" ? false : result2 === 0;
    };
  }
};
function isLL1Sequence(sequences, allowEmpty = true) {
  const fullSet = /* @__PURE__ */ new Set();
  for (const alt of sequences) {
    const altSet = /* @__PURE__ */ new Set();
    for (const tokType of alt) {
      if (tokType === void 0) {
        if (allowEmpty) {
          break;
        } else {
          return false;
        }
      }
      const indices = [tokType.tokenTypeIdx].concat(tokType.categoryMatches);
      for (const index of indices) {
        if (fullSet.has(index)) {
          if (!altSet.has(index)) {
            return false;
          }
        } else {
          fullSet.add(index);
          altSet.add(index);
        }
      }
    }
  }
  return true;
}
function initATNSimulator(atn) {
  const decisionLength = atn.decisionStates.length;
  const decisionToDFA = Array(decisionLength);
  for (let i = 0; i < decisionLength; i++) {
    decisionToDFA[i] = createDFACache(atn.decisionStates[i], i);
  }
  return decisionToDFA;
}
function adaptivePredict(dfaCaches, decision, predicateSet, logging) {
  const dfa = dfaCaches[decision](predicateSet);
  let start = dfa.start;
  if (start === void 0) {
    const closure2 = computeStartState(dfa.atnStartState);
    start = addDFAState(dfa, newDFAState(closure2));
    dfa.start = start;
  }
  const alt = performLookahead.apply(this, [dfa, start, predicateSet, logging]);
  return alt;
}
function performLookahead(dfa, s0, predicateSet, logging) {
  let previousD = s0;
  let i = 1;
  const path = [];
  let t = this.LA(i++);
  while (true) {
    let d = getExistingTargetState(previousD, t);
    if (d === void 0) {
      d = computeLookaheadTarget.apply(this, [dfa, previousD, t, i, predicateSet, logging]);
    }
    if (d === DFA_ERROR) {
      return buildAdaptivePredictError(path, previousD, t);
    }
    if (d.isAcceptState === true) {
      return d.prediction;
    }
    previousD = d;
    path.push(t);
    t = this.LA(i++);
  }
}
function computeLookaheadTarget(dfa, previousD, token, lookahead, predicateSet, logging) {
  const reach = computeReachSet(previousD.configs, token, predicateSet);
  if (reach.size === 0) {
    addDFAEdge(dfa, previousD, token, DFA_ERROR);
    return DFA_ERROR;
  }
  let newState2 = newDFAState(reach);
  const predictedAlt = getUniqueAlt(reach, predicateSet);
  if (predictedAlt !== void 0) {
    newState2.isAcceptState = true;
    newState2.prediction = predictedAlt;
    newState2.configs.uniqueAlt = predictedAlt;
  } else if (hasConflictTerminatingPrediction(reach)) {
    const prediction = min_default(reach.alts);
    newState2.isAcceptState = true;
    newState2.prediction = prediction;
    newState2.configs.uniqueAlt = prediction;
    reportLookaheadAmbiguity.apply(this, [dfa, lookahead, reach.alts, logging]);
  }
  newState2 = addDFAEdge(dfa, previousD, token, newState2);
  return newState2;
}
function reportLookaheadAmbiguity(dfa, lookahead, ambiguityIndices, logging) {
  const prefixPath = [];
  for (let i = 1; i <= lookahead; i++) {
    prefixPath.push(this.LA(i).tokenType);
  }
  const atnState = dfa.atnStartState;
  const topLevelRule = atnState.rule;
  const production = atnState.production;
  const message = buildAmbiguityError({
    topLevelRule,
    ambiguityIndices,
    production,
    prefixPath
  });
  logging(message);
}
function buildAmbiguityError(options) {
  const pathMsg = map_default(options.prefixPath, (currtok) => tokenLabel2(currtok)).join(", ");
  const occurrence = options.production.idx === 0 ? "" : options.production.idx;
  let currMessage = `Ambiguous Alternatives Detected: <${options.ambiguityIndices.join(", ")}> in <${getProductionDslName2(options.production)}${occurrence}> inside <${options.topLevelRule.name}> Rule,
<${pathMsg}> may appears as a prefix path in all these alternatives.
`;
  currMessage = currMessage + `See: https://chevrotain.io/docs/guide/resolving_grammar_errors.html#AMBIGUOUS_ALTERNATIVES
For Further details.`;
  return currMessage;
}
function getProductionDslName2(prod) {
  if (prod instanceof NonTerminal) {
    return "SUBRULE";
  } else if (prod instanceof Option) {
    return "OPTION";
  } else if (prod instanceof Alternation) {
    return "OR";
  } else if (prod instanceof RepetitionMandatory) {
    return "AT_LEAST_ONE";
  } else if (prod instanceof RepetitionMandatoryWithSeparator) {
    return "AT_LEAST_ONE_SEP";
  } else if (prod instanceof RepetitionWithSeparator) {
    return "MANY_SEP";
  } else if (prod instanceof Repetition) {
    return "MANY";
  } else if (prod instanceof Terminal) {
    return "CONSUME";
  } else {
    throw Error("non exhaustive match");
  }
}
function buildAdaptivePredictError(path, previous, current) {
  const nextTransitions = flatMap_default(previous.configs.elements, (e) => e.state.transitions);
  const nextTokenTypes = uniqBy_default(nextTransitions.filter((e) => e instanceof AtomTransition).map((e) => e.tokenType), (e) => e.tokenTypeIdx);
  return {
    actualToken: current,
    possibleTokenTypes: nextTokenTypes,
    tokenPath: path
  };
}
function getExistingTargetState(state, token) {
  return state.edges[token.tokenTypeIdx];
}
function computeReachSet(configs, token, predicateSet) {
  const intermediate = new ATNConfigSet();
  const skippedStopStates = [];
  for (const c of configs.elements) {
    if (predicateSet.is(c.alt) === false) {
      continue;
    }
    if (c.state.type === ATN_RULE_STOP) {
      skippedStopStates.push(c);
      continue;
    }
    const transitionLength = c.state.transitions.length;
    for (let i = 0; i < transitionLength; i++) {
      const transition = c.state.transitions[i];
      const target = getReachableTarget(transition, token);
      if (target !== void 0) {
        intermediate.add({
          state: target,
          alt: c.alt,
          stack: c.stack
        });
      }
    }
  }
  let reach;
  if (skippedStopStates.length === 0 && intermediate.size === 1) {
    reach = intermediate;
  }
  if (reach === void 0) {
    reach = new ATNConfigSet();
    for (const c of intermediate.elements) {
      closure(c, reach);
    }
  }
  if (skippedStopStates.length > 0 && !hasConfigInRuleStopState(reach)) {
    for (const c of skippedStopStates) {
      reach.add(c);
    }
  }
  return reach;
}
function getReachableTarget(transition, token) {
  if (transition instanceof AtomTransition && tokenMatcher(token, transition.tokenType)) {
    return transition.target;
  }
  return void 0;
}
function getUniqueAlt(configs, predicateSet) {
  let alt;
  for (const c of configs.elements) {
    if (predicateSet.is(c.alt) === true) {
      if (alt === void 0) {
        alt = c.alt;
      } else if (alt !== c.alt) {
        return void 0;
      }
    }
  }
  return alt;
}
function newDFAState(closure2) {
  return {
    configs: closure2,
    edges: {},
    isAcceptState: false,
    prediction: -1
  };
}
function addDFAEdge(dfa, from, token, to) {
  to = addDFAState(dfa, to);
  from.edges[token.tokenTypeIdx] = to;
  return to;
}
function addDFAState(dfa, state) {
  if (state === DFA_ERROR) {
    return state;
  }
  const mapKey = state.configs.key;
  const existing = dfa.states[mapKey];
  if (existing !== void 0) {
    return existing;
  }
  state.configs.finalize();
  dfa.states[mapKey] = state;
  return state;
}
function computeStartState(atnState) {
  const configs = new ATNConfigSet();
  const numberOfTransitions = atnState.transitions.length;
  for (let i = 0; i < numberOfTransitions; i++) {
    const target = atnState.transitions[i].target;
    const config = {
      state: target,
      alt: i,
      stack: []
    };
    closure(config, configs);
  }
  return configs;
}
function closure(config, configs) {
  const p = config.state;
  if (p.type === ATN_RULE_STOP) {
    if (config.stack.length > 0) {
      const atnStack = [...config.stack];
      const followState = atnStack.pop();
      const followConfig = {
        state: followState,
        alt: config.alt,
        stack: atnStack
      };
      closure(followConfig, configs);
    } else {
      configs.add(config);
    }
    return;
  }
  if (!p.epsilonOnlyTransitions) {
    configs.add(config);
  }
  const transitionLength = p.transitions.length;
  for (let i = 0; i < transitionLength; i++) {
    const transition = p.transitions[i];
    const c = getEpsilonTarget(config, transition);
    if (c !== void 0) {
      closure(c, configs);
    }
  }
}
function getEpsilonTarget(config, transition) {
  if (transition instanceof EpsilonTransition) {
    return {
      state: transition.target,
      alt: config.alt,
      stack: config.stack
    };
  } else if (transition instanceof RuleTransition) {
    const stack = [...config.stack, transition.followState];
    return {
      state: transition.target,
      alt: config.alt,
      stack
    };
  }
  return void 0;
}
function hasConfigInRuleStopState(configs) {
  for (const c of configs.elements) {
    if (c.state.type === ATN_RULE_STOP) {
      return true;
    }
  }
  return false;
}
function allConfigsInRuleStopStates(configs) {
  for (const c of configs.elements) {
    if (c.state.type !== ATN_RULE_STOP) {
      return false;
    }
  }
  return true;
}
function hasConflictTerminatingPrediction(configs) {
  if (allConfigsInRuleStopStates(configs)) {
    return true;
  }
  const altSets = getConflictingAltSets(configs.elements);
  const heuristic = hasConflictingAltSet(altSets) && !hasStateAssociatedWithOneAlt(altSets);
  return heuristic;
}
function getConflictingAltSets(configs) {
  const configToAlts = /* @__PURE__ */ new Map();
  for (const c of configs) {
    const key = getATNConfigKey(c, false);
    let alts = configToAlts.get(key);
    if (alts === void 0) {
      alts = {};
      configToAlts.set(key, alts);
    }
    alts[c.alt] = true;
  }
  return configToAlts;
}
function hasConflictingAltSet(altSets) {
  for (const value of Array.from(altSets.values())) {
    if (Object.keys(value).length > 1) {
      return true;
    }
  }
  return false;
}
function hasStateAssociatedWithOneAlt(altSets) {
  for (const value of Array.from(altSets.values())) {
    if (Object.keys(value).length === 1) {
      return true;
    }
  }
  return false;
}

// node_modules/vscode-languageserver-types/lib/esm/main.js
var DocumentUri;
(function(DocumentUri2) {
  function is(value) {
    return typeof value === "string";
  }
  DocumentUri2.is = is;
})(DocumentUri || (DocumentUri = {}));
var URI;
(function(URI3) {
  function is(value) {
    return typeof value === "string";
  }
  URI3.is = is;
})(URI || (URI = {}));
var integer;
(function(integer2) {
  integer2.MIN_VALUE = -2147483648;
  integer2.MAX_VALUE = 2147483647;
  function is(value) {
    return typeof value === "number" && integer2.MIN_VALUE <= value && value <= integer2.MAX_VALUE;
  }
  integer2.is = is;
})(integer || (integer = {}));
var uinteger;
(function(uinteger2) {
  uinteger2.MIN_VALUE = 0;
  uinteger2.MAX_VALUE = 2147483647;
  function is(value) {
    return typeof value === "number" && uinteger2.MIN_VALUE <= value && value <= uinteger2.MAX_VALUE;
  }
  uinteger2.is = is;
})(uinteger || (uinteger = {}));
var Position;
(function(Position2) {
  function create2(line, character) {
    if (line === Number.MAX_VALUE) {
      line = uinteger.MAX_VALUE;
    }
    if (character === Number.MAX_VALUE) {
      character = uinteger.MAX_VALUE;
    }
    return { line, character };
  }
  Position2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.objectLiteral(candidate) && Is.uinteger(candidate.line) && Is.uinteger(candidate.character);
  }
  Position2.is = is;
})(Position || (Position = {}));
var Range;
(function(Range2) {
  function create2(one, two, three, four) {
    if (Is.uinteger(one) && Is.uinteger(two) && Is.uinteger(three) && Is.uinteger(four)) {
      return { start: Position.create(one, two), end: Position.create(three, four) };
    } else if (Position.is(one) && Position.is(two)) {
      return { start: one, end: two };
    } else {
      throw new Error(`Range#create called with invalid arguments[${one}, ${two}, ${three}, ${four}]`);
    }
  }
  Range2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.objectLiteral(candidate) && Position.is(candidate.start) && Position.is(candidate.end);
  }
  Range2.is = is;
})(Range || (Range = {}));
var Location;
(function(Location2) {
  function create2(uri, range2) {
    return { uri, range: range2 };
  }
  Location2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.objectLiteral(candidate) && Range.is(candidate.range) && (Is.string(candidate.uri) || Is.undefined(candidate.uri));
  }
  Location2.is = is;
})(Location || (Location = {}));
var LocationLink;
(function(LocationLink2) {
  function create2(targetUri, targetRange, targetSelectionRange, originSelectionRange) {
    return { targetUri, targetRange, targetSelectionRange, originSelectionRange };
  }
  LocationLink2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.objectLiteral(candidate) && Range.is(candidate.targetRange) && Is.string(candidate.targetUri) && Range.is(candidate.targetSelectionRange) && (Range.is(candidate.originSelectionRange) || Is.undefined(candidate.originSelectionRange));
  }
  LocationLink2.is = is;
})(LocationLink || (LocationLink = {}));
var Color;
(function(Color2) {
  function create2(red, green, blue, alpha) {
    return {
      red,
      green,
      blue,
      alpha
    };
  }
  Color2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Is.numberRange(candidate.red, 0, 1) && Is.numberRange(candidate.green, 0, 1) && Is.numberRange(candidate.blue, 0, 1) && Is.numberRange(candidate.alpha, 0, 1);
  }
  Color2.is = is;
})(Color || (Color = {}));
var ColorInformation;
(function(ColorInformation2) {
  function create2(range2, color) {
    return {
      range: range2,
      color
    };
  }
  ColorInformation2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Range.is(candidate.range) && Color.is(candidate.color);
  }
  ColorInformation2.is = is;
})(ColorInformation || (ColorInformation = {}));
var ColorPresentation;
(function(ColorPresentation2) {
  function create2(label, textEdit, additionalTextEdits) {
    return {
      label,
      textEdit,
      additionalTextEdits
    };
  }
  ColorPresentation2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Is.string(candidate.label) && (Is.undefined(candidate.textEdit) || TextEdit.is(candidate)) && (Is.undefined(candidate.additionalTextEdits) || Is.typedArray(candidate.additionalTextEdits, TextEdit.is));
  }
  ColorPresentation2.is = is;
})(ColorPresentation || (ColorPresentation = {}));
var FoldingRangeKind;
(function(FoldingRangeKind2) {
  FoldingRangeKind2.Comment = "comment";
  FoldingRangeKind2.Imports = "imports";
  FoldingRangeKind2.Region = "region";
})(FoldingRangeKind || (FoldingRangeKind = {}));
var FoldingRange;
(function(FoldingRange2) {
  function create2(startLine, endLine, startCharacter, endCharacter, kind, collapsedText) {
    const result2 = {
      startLine,
      endLine
    };
    if (Is.defined(startCharacter)) {
      result2.startCharacter = startCharacter;
    }
    if (Is.defined(endCharacter)) {
      result2.endCharacter = endCharacter;
    }
    if (Is.defined(kind)) {
      result2.kind = kind;
    }
    if (Is.defined(collapsedText)) {
      result2.collapsedText = collapsedText;
    }
    return result2;
  }
  FoldingRange2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Is.uinteger(candidate.startLine) && Is.uinteger(candidate.startLine) && (Is.undefined(candidate.startCharacter) || Is.uinteger(candidate.startCharacter)) && (Is.undefined(candidate.endCharacter) || Is.uinteger(candidate.endCharacter)) && (Is.undefined(candidate.kind) || Is.string(candidate.kind));
  }
  FoldingRange2.is = is;
})(FoldingRange || (FoldingRange = {}));
var DiagnosticRelatedInformation;
(function(DiagnosticRelatedInformation2) {
  function create2(location, message) {
    return {
      location,
      message
    };
  }
  DiagnosticRelatedInformation2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Location.is(candidate.location) && Is.string(candidate.message);
  }
  DiagnosticRelatedInformation2.is = is;
})(DiagnosticRelatedInformation || (DiagnosticRelatedInformation = {}));
var DiagnosticSeverity;
(function(DiagnosticSeverity2) {
  DiagnosticSeverity2.Error = 1;
  DiagnosticSeverity2.Warning = 2;
  DiagnosticSeverity2.Information = 3;
  DiagnosticSeverity2.Hint = 4;
})(DiagnosticSeverity || (DiagnosticSeverity = {}));
var DiagnosticTag;
(function(DiagnosticTag2) {
  DiagnosticTag2.Unnecessary = 1;
  DiagnosticTag2.Deprecated = 2;
})(DiagnosticTag || (DiagnosticTag = {}));
var CodeDescription;
(function(CodeDescription2) {
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Is.string(candidate.href);
  }
  CodeDescription2.is = is;
})(CodeDescription || (CodeDescription = {}));
var Diagnostic;
(function(Diagnostic2) {
  function create2(range2, message, severity, code, source, relatedInformation) {
    let result2 = { range: range2, message };
    if (Is.defined(severity)) {
      result2.severity = severity;
    }
    if (Is.defined(code)) {
      result2.code = code;
    }
    if (Is.defined(source)) {
      result2.source = source;
    }
    if (Is.defined(relatedInformation)) {
      result2.relatedInformation = relatedInformation;
    }
    return result2;
  }
  Diagnostic2.create = create2;
  function is(value) {
    var _a;
    let candidate = value;
    return Is.defined(candidate) && Range.is(candidate.range) && Is.string(candidate.message) && (Is.number(candidate.severity) || Is.undefined(candidate.severity)) && (Is.integer(candidate.code) || Is.string(candidate.code) || Is.undefined(candidate.code)) && (Is.undefined(candidate.codeDescription) || Is.string((_a = candidate.codeDescription) === null || _a === void 0 ? void 0 : _a.href)) && (Is.string(candidate.source) || Is.undefined(candidate.source)) && (Is.undefined(candidate.relatedInformation) || Is.typedArray(candidate.relatedInformation, DiagnosticRelatedInformation.is));
  }
  Diagnostic2.is = is;
})(Diagnostic || (Diagnostic = {}));
var Command;
(function(Command2) {
  function create2(title, command, ...args) {
    let result2 = { title, command };
    if (Is.defined(args) && args.length > 0) {
      result2.arguments = args;
    }
    return result2;
  }
  Command2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.string(candidate.title) && Is.string(candidate.command);
  }
  Command2.is = is;
})(Command || (Command = {}));
var TextEdit;
(function(TextEdit2) {
  function replace2(range2, newText) {
    return { range: range2, newText };
  }
  TextEdit2.replace = replace2;
  function insert(position, newText) {
    return { range: { start: position, end: position }, newText };
  }
  TextEdit2.insert = insert;
  function del(range2) {
    return { range: range2, newText: "" };
  }
  TextEdit2.del = del;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Is.string(candidate.newText) && Range.is(candidate.range);
  }
  TextEdit2.is = is;
})(TextEdit || (TextEdit = {}));
var ChangeAnnotation;
(function(ChangeAnnotation2) {
  function create2(label, needsConfirmation, description) {
    const result2 = { label };
    if (needsConfirmation !== void 0) {
      result2.needsConfirmation = needsConfirmation;
    }
    if (description !== void 0) {
      result2.description = description;
    }
    return result2;
  }
  ChangeAnnotation2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Is.string(candidate.label) && (Is.boolean(candidate.needsConfirmation) || candidate.needsConfirmation === void 0) && (Is.string(candidate.description) || candidate.description === void 0);
  }
  ChangeAnnotation2.is = is;
})(ChangeAnnotation || (ChangeAnnotation = {}));
var ChangeAnnotationIdentifier;
(function(ChangeAnnotationIdentifier2) {
  function is(value) {
    const candidate = value;
    return Is.string(candidate);
  }
  ChangeAnnotationIdentifier2.is = is;
})(ChangeAnnotationIdentifier || (ChangeAnnotationIdentifier = {}));
var AnnotatedTextEdit;
(function(AnnotatedTextEdit2) {
  function replace2(range2, newText, annotation) {
    return { range: range2, newText, annotationId: annotation };
  }
  AnnotatedTextEdit2.replace = replace2;
  function insert(position, newText, annotation) {
    return { range: { start: position, end: position }, newText, annotationId: annotation };
  }
  AnnotatedTextEdit2.insert = insert;
  function del(range2, annotation) {
    return { range: range2, newText: "", annotationId: annotation };
  }
  AnnotatedTextEdit2.del = del;
  function is(value) {
    const candidate = value;
    return TextEdit.is(candidate) && (ChangeAnnotation.is(candidate.annotationId) || ChangeAnnotationIdentifier.is(candidate.annotationId));
  }
  AnnotatedTextEdit2.is = is;
})(AnnotatedTextEdit || (AnnotatedTextEdit = {}));
var TextDocumentEdit;
(function(TextDocumentEdit2) {
  function create2(textDocument, edits) {
    return { textDocument, edits };
  }
  TextDocumentEdit2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && OptionalVersionedTextDocumentIdentifier.is(candidate.textDocument) && Array.isArray(candidate.edits);
  }
  TextDocumentEdit2.is = is;
})(TextDocumentEdit || (TextDocumentEdit = {}));
var CreateFile;
(function(CreateFile2) {
  function create2(uri, options, annotation) {
    let result2 = {
      kind: "create",
      uri
    };
    if (options !== void 0 && (options.overwrite !== void 0 || options.ignoreIfExists !== void 0)) {
      result2.options = options;
    }
    if (annotation !== void 0) {
      result2.annotationId = annotation;
    }
    return result2;
  }
  CreateFile2.create = create2;
  function is(value) {
    let candidate = value;
    return candidate && candidate.kind === "create" && Is.string(candidate.uri) && (candidate.options === void 0 || (candidate.options.overwrite === void 0 || Is.boolean(candidate.options.overwrite)) && (candidate.options.ignoreIfExists === void 0 || Is.boolean(candidate.options.ignoreIfExists))) && (candidate.annotationId === void 0 || ChangeAnnotationIdentifier.is(candidate.annotationId));
  }
  CreateFile2.is = is;
})(CreateFile || (CreateFile = {}));
var RenameFile;
(function(RenameFile2) {
  function create2(oldUri, newUri, options, annotation) {
    let result2 = {
      kind: "rename",
      oldUri,
      newUri
    };
    if (options !== void 0 && (options.overwrite !== void 0 || options.ignoreIfExists !== void 0)) {
      result2.options = options;
    }
    if (annotation !== void 0) {
      result2.annotationId = annotation;
    }
    return result2;
  }
  RenameFile2.create = create2;
  function is(value) {
    let candidate = value;
    return candidate && candidate.kind === "rename" && Is.string(candidate.oldUri) && Is.string(candidate.newUri) && (candidate.options === void 0 || (candidate.options.overwrite === void 0 || Is.boolean(candidate.options.overwrite)) && (candidate.options.ignoreIfExists === void 0 || Is.boolean(candidate.options.ignoreIfExists))) && (candidate.annotationId === void 0 || ChangeAnnotationIdentifier.is(candidate.annotationId));
  }
  RenameFile2.is = is;
})(RenameFile || (RenameFile = {}));
var DeleteFile;
(function(DeleteFile2) {
  function create2(uri, options, annotation) {
    let result2 = {
      kind: "delete",
      uri
    };
    if (options !== void 0 && (options.recursive !== void 0 || options.ignoreIfNotExists !== void 0)) {
      result2.options = options;
    }
    if (annotation !== void 0) {
      result2.annotationId = annotation;
    }
    return result2;
  }
  DeleteFile2.create = create2;
  function is(value) {
    let candidate = value;
    return candidate && candidate.kind === "delete" && Is.string(candidate.uri) && (candidate.options === void 0 || (candidate.options.recursive === void 0 || Is.boolean(candidate.options.recursive)) && (candidate.options.ignoreIfNotExists === void 0 || Is.boolean(candidate.options.ignoreIfNotExists))) && (candidate.annotationId === void 0 || ChangeAnnotationIdentifier.is(candidate.annotationId));
  }
  DeleteFile2.is = is;
})(DeleteFile || (DeleteFile = {}));
var WorkspaceEdit;
(function(WorkspaceEdit2) {
  function is(value) {
    let candidate = value;
    return candidate && (candidate.changes !== void 0 || candidate.documentChanges !== void 0) && (candidate.documentChanges === void 0 || candidate.documentChanges.every((change) => {
      if (Is.string(change.kind)) {
        return CreateFile.is(change) || RenameFile.is(change) || DeleteFile.is(change);
      } else {
        return TextDocumentEdit.is(change);
      }
    }));
  }
  WorkspaceEdit2.is = is;
})(WorkspaceEdit || (WorkspaceEdit = {}));
var TextDocumentIdentifier;
(function(TextDocumentIdentifier2) {
  function create2(uri) {
    return { uri };
  }
  TextDocumentIdentifier2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.string(candidate.uri);
  }
  TextDocumentIdentifier2.is = is;
})(TextDocumentIdentifier || (TextDocumentIdentifier = {}));
var VersionedTextDocumentIdentifier;
(function(VersionedTextDocumentIdentifier2) {
  function create2(uri, version) {
    return { uri, version };
  }
  VersionedTextDocumentIdentifier2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.string(candidate.uri) && Is.integer(candidate.version);
  }
  VersionedTextDocumentIdentifier2.is = is;
})(VersionedTextDocumentIdentifier || (VersionedTextDocumentIdentifier = {}));
var OptionalVersionedTextDocumentIdentifier;
(function(OptionalVersionedTextDocumentIdentifier2) {
  function create2(uri, version) {
    return { uri, version };
  }
  OptionalVersionedTextDocumentIdentifier2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.string(candidate.uri) && (candidate.version === null || Is.integer(candidate.version));
  }
  OptionalVersionedTextDocumentIdentifier2.is = is;
})(OptionalVersionedTextDocumentIdentifier || (OptionalVersionedTextDocumentIdentifier = {}));
var TextDocumentItem;
(function(TextDocumentItem2) {
  function create2(uri, languageId, version, text) {
    return { uri, languageId, version, text };
  }
  TextDocumentItem2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.string(candidate.uri) && Is.string(candidate.languageId) && Is.integer(candidate.version) && Is.string(candidate.text);
  }
  TextDocumentItem2.is = is;
})(TextDocumentItem || (TextDocumentItem = {}));
var MarkupKind;
(function(MarkupKind2) {
  MarkupKind2.PlainText = "plaintext";
  MarkupKind2.Markdown = "markdown";
  function is(value) {
    const candidate = value;
    return candidate === MarkupKind2.PlainText || candidate === MarkupKind2.Markdown;
  }
  MarkupKind2.is = is;
})(MarkupKind || (MarkupKind = {}));
var MarkupContent;
(function(MarkupContent2) {
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(value) && MarkupKind.is(candidate.kind) && Is.string(candidate.value);
  }
  MarkupContent2.is = is;
})(MarkupContent || (MarkupContent = {}));
var CompletionItemKind;
(function(CompletionItemKind2) {
  CompletionItemKind2.Text = 1;
  CompletionItemKind2.Method = 2;
  CompletionItemKind2.Function = 3;
  CompletionItemKind2.Constructor = 4;
  CompletionItemKind2.Field = 5;
  CompletionItemKind2.Variable = 6;
  CompletionItemKind2.Class = 7;
  CompletionItemKind2.Interface = 8;
  CompletionItemKind2.Module = 9;
  CompletionItemKind2.Property = 10;
  CompletionItemKind2.Unit = 11;
  CompletionItemKind2.Value = 12;
  CompletionItemKind2.Enum = 13;
  CompletionItemKind2.Keyword = 14;
  CompletionItemKind2.Snippet = 15;
  CompletionItemKind2.Color = 16;
  CompletionItemKind2.File = 17;
  CompletionItemKind2.Reference = 18;
  CompletionItemKind2.Folder = 19;
  CompletionItemKind2.EnumMember = 20;
  CompletionItemKind2.Constant = 21;
  CompletionItemKind2.Struct = 22;
  CompletionItemKind2.Event = 23;
  CompletionItemKind2.Operator = 24;
  CompletionItemKind2.TypeParameter = 25;
})(CompletionItemKind || (CompletionItemKind = {}));
var InsertTextFormat;
(function(InsertTextFormat2) {
  InsertTextFormat2.PlainText = 1;
  InsertTextFormat2.Snippet = 2;
})(InsertTextFormat || (InsertTextFormat = {}));
var CompletionItemTag;
(function(CompletionItemTag2) {
  CompletionItemTag2.Deprecated = 1;
})(CompletionItemTag || (CompletionItemTag = {}));
var InsertReplaceEdit;
(function(InsertReplaceEdit2) {
  function create2(newText, insert, replace2) {
    return { newText, insert, replace: replace2 };
  }
  InsertReplaceEdit2.create = create2;
  function is(value) {
    const candidate = value;
    return candidate && Is.string(candidate.newText) && Range.is(candidate.insert) && Range.is(candidate.replace);
  }
  InsertReplaceEdit2.is = is;
})(InsertReplaceEdit || (InsertReplaceEdit = {}));
var InsertTextMode;
(function(InsertTextMode2) {
  InsertTextMode2.asIs = 1;
  InsertTextMode2.adjustIndentation = 2;
})(InsertTextMode || (InsertTextMode = {}));
var CompletionItemLabelDetails;
(function(CompletionItemLabelDetails2) {
  function is(value) {
    const candidate = value;
    return candidate && (Is.string(candidate.detail) || candidate.detail === void 0) && (Is.string(candidate.description) || candidate.description === void 0);
  }
  CompletionItemLabelDetails2.is = is;
})(CompletionItemLabelDetails || (CompletionItemLabelDetails = {}));
var CompletionItem;
(function(CompletionItem2) {
  function create2(label) {
    return { label };
  }
  CompletionItem2.create = create2;
})(CompletionItem || (CompletionItem = {}));
var CompletionList;
(function(CompletionList2) {
  function create2(items, isIncomplete) {
    return { items: items ? items : [], isIncomplete: !!isIncomplete };
  }
  CompletionList2.create = create2;
})(CompletionList || (CompletionList = {}));
var MarkedString;
(function(MarkedString2) {
  function fromPlainText(plainText) {
    return plainText.replace(/[\\`*_{}[\]()#+\-.!]/g, "\\$&");
  }
  MarkedString2.fromPlainText = fromPlainText;
  function is(value) {
    const candidate = value;
    return Is.string(candidate) || Is.objectLiteral(candidate) && Is.string(candidate.language) && Is.string(candidate.value);
  }
  MarkedString2.is = is;
})(MarkedString || (MarkedString = {}));
var Hover;
(function(Hover2) {
  function is(value) {
    let candidate = value;
    return !!candidate && Is.objectLiteral(candidate) && (MarkupContent.is(candidate.contents) || MarkedString.is(candidate.contents) || Is.typedArray(candidate.contents, MarkedString.is)) && (value.range === void 0 || Range.is(value.range));
  }
  Hover2.is = is;
})(Hover || (Hover = {}));
var ParameterInformation;
(function(ParameterInformation2) {
  function create2(label, documentation) {
    return documentation ? { label, documentation } : { label };
  }
  ParameterInformation2.create = create2;
})(ParameterInformation || (ParameterInformation = {}));
var SignatureInformation;
(function(SignatureInformation2) {
  function create2(label, documentation, ...parameters) {
    let result2 = { label };
    if (Is.defined(documentation)) {
      result2.documentation = documentation;
    }
    if (Is.defined(parameters)) {
      result2.parameters = parameters;
    } else {
      result2.parameters = [];
    }
    return result2;
  }
  SignatureInformation2.create = create2;
})(SignatureInformation || (SignatureInformation = {}));
var DocumentHighlightKind;
(function(DocumentHighlightKind2) {
  DocumentHighlightKind2.Text = 1;
  DocumentHighlightKind2.Read = 2;
  DocumentHighlightKind2.Write = 3;
})(DocumentHighlightKind || (DocumentHighlightKind = {}));
var DocumentHighlight;
(function(DocumentHighlight2) {
  function create2(range2, kind) {
    let result2 = { range: range2 };
    if (Is.number(kind)) {
      result2.kind = kind;
    }
    return result2;
  }
  DocumentHighlight2.create = create2;
})(DocumentHighlight || (DocumentHighlight = {}));
var SymbolKind;
(function(SymbolKind2) {
  SymbolKind2.File = 1;
  SymbolKind2.Module = 2;
  SymbolKind2.Namespace = 3;
  SymbolKind2.Package = 4;
  SymbolKind2.Class = 5;
  SymbolKind2.Method = 6;
  SymbolKind2.Property = 7;
  SymbolKind2.Field = 8;
  SymbolKind2.Constructor = 9;
  SymbolKind2.Enum = 10;
  SymbolKind2.Interface = 11;
  SymbolKind2.Function = 12;
  SymbolKind2.Variable = 13;
  SymbolKind2.Constant = 14;
  SymbolKind2.String = 15;
  SymbolKind2.Number = 16;
  SymbolKind2.Boolean = 17;
  SymbolKind2.Array = 18;
  SymbolKind2.Object = 19;
  SymbolKind2.Key = 20;
  SymbolKind2.Null = 21;
  SymbolKind2.EnumMember = 22;
  SymbolKind2.Struct = 23;
  SymbolKind2.Event = 24;
  SymbolKind2.Operator = 25;
  SymbolKind2.TypeParameter = 26;
})(SymbolKind || (SymbolKind = {}));
var SymbolTag;
(function(SymbolTag2) {
  SymbolTag2.Deprecated = 1;
})(SymbolTag || (SymbolTag = {}));
var SymbolInformation;
(function(SymbolInformation2) {
  function create2(name, kind, range2, uri, containerName) {
    let result2 = {
      name,
      kind,
      location: { uri, range: range2 }
    };
    if (containerName) {
      result2.containerName = containerName;
    }
    return result2;
  }
  SymbolInformation2.create = create2;
})(SymbolInformation || (SymbolInformation = {}));
var WorkspaceSymbol;
(function(WorkspaceSymbol2) {
  function create2(name, kind, uri, range2) {
    return range2 !== void 0 ? { name, kind, location: { uri, range: range2 } } : { name, kind, location: { uri } };
  }
  WorkspaceSymbol2.create = create2;
})(WorkspaceSymbol || (WorkspaceSymbol = {}));
var DocumentSymbol;
(function(DocumentSymbol2) {
  function create2(name, detail, kind, range2, selectionRange, children) {
    let result2 = {
      name,
      detail,
      kind,
      range: range2,
      selectionRange
    };
    if (children !== void 0) {
      result2.children = children;
    }
    return result2;
  }
  DocumentSymbol2.create = create2;
  function is(value) {
    let candidate = value;
    return candidate && Is.string(candidate.name) && Is.number(candidate.kind) && Range.is(candidate.range) && Range.is(candidate.selectionRange) && (candidate.detail === void 0 || Is.string(candidate.detail)) && (candidate.deprecated === void 0 || Is.boolean(candidate.deprecated)) && (candidate.children === void 0 || Array.isArray(candidate.children)) && (candidate.tags === void 0 || Array.isArray(candidate.tags));
  }
  DocumentSymbol2.is = is;
})(DocumentSymbol || (DocumentSymbol = {}));
var CodeActionKind;
(function(CodeActionKind2) {
  CodeActionKind2.Empty = "";
  CodeActionKind2.QuickFix = "quickfix";
  CodeActionKind2.Refactor = "refactor";
  CodeActionKind2.RefactorExtract = "refactor.extract";
  CodeActionKind2.RefactorInline = "refactor.inline";
  CodeActionKind2.RefactorRewrite = "refactor.rewrite";
  CodeActionKind2.Source = "source";
  CodeActionKind2.SourceOrganizeImports = "source.organizeImports";
  CodeActionKind2.SourceFixAll = "source.fixAll";
})(CodeActionKind || (CodeActionKind = {}));
var CodeActionTriggerKind;
(function(CodeActionTriggerKind2) {
  CodeActionTriggerKind2.Invoked = 1;
  CodeActionTriggerKind2.Automatic = 2;
})(CodeActionTriggerKind || (CodeActionTriggerKind = {}));
var CodeActionContext;
(function(CodeActionContext2) {
  function create2(diagnostics, only, triggerKind) {
    let result2 = { diagnostics };
    if (only !== void 0 && only !== null) {
      result2.only = only;
    }
    if (triggerKind !== void 0 && triggerKind !== null) {
      result2.triggerKind = triggerKind;
    }
    return result2;
  }
  CodeActionContext2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.typedArray(candidate.diagnostics, Diagnostic.is) && (candidate.only === void 0 || Is.typedArray(candidate.only, Is.string)) && (candidate.triggerKind === void 0 || candidate.triggerKind === CodeActionTriggerKind.Invoked || candidate.triggerKind === CodeActionTriggerKind.Automatic);
  }
  CodeActionContext2.is = is;
})(CodeActionContext || (CodeActionContext = {}));
var CodeAction;
(function(CodeAction2) {
  function create2(title, kindOrCommandOrEdit, kind) {
    let result2 = { title };
    let checkKind = true;
    if (typeof kindOrCommandOrEdit === "string") {
      checkKind = false;
      result2.kind = kindOrCommandOrEdit;
    } else if (Command.is(kindOrCommandOrEdit)) {
      result2.command = kindOrCommandOrEdit;
    } else {
      result2.edit = kindOrCommandOrEdit;
    }
    if (checkKind && kind !== void 0) {
      result2.kind = kind;
    }
    return result2;
  }
  CodeAction2.create = create2;
  function is(value) {
    let candidate = value;
    return candidate && Is.string(candidate.title) && (candidate.diagnostics === void 0 || Is.typedArray(candidate.diagnostics, Diagnostic.is)) && (candidate.kind === void 0 || Is.string(candidate.kind)) && (candidate.edit !== void 0 || candidate.command !== void 0) && (candidate.command === void 0 || Command.is(candidate.command)) && (candidate.isPreferred === void 0 || Is.boolean(candidate.isPreferred)) && (candidate.edit === void 0 || WorkspaceEdit.is(candidate.edit));
  }
  CodeAction2.is = is;
})(CodeAction || (CodeAction = {}));
var CodeLens;
(function(CodeLens2) {
  function create2(range2, data) {
    let result2 = { range: range2 };
    if (Is.defined(data)) {
      result2.data = data;
    }
    return result2;
  }
  CodeLens2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Range.is(candidate.range) && (Is.undefined(candidate.command) || Command.is(candidate.command));
  }
  CodeLens2.is = is;
})(CodeLens || (CodeLens = {}));
var FormattingOptions;
(function(FormattingOptions2) {
  function create2(tabSize, insertSpaces) {
    return { tabSize, insertSpaces };
  }
  FormattingOptions2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.uinteger(candidate.tabSize) && Is.boolean(candidate.insertSpaces);
  }
  FormattingOptions2.is = is;
})(FormattingOptions || (FormattingOptions = {}));
var DocumentLink;
(function(DocumentLink2) {
  function create2(range2, target, data) {
    return { range: range2, target, data };
  }
  DocumentLink2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Range.is(candidate.range) && (Is.undefined(candidate.target) || Is.string(candidate.target));
  }
  DocumentLink2.is = is;
})(DocumentLink || (DocumentLink = {}));
var SelectionRange;
(function(SelectionRange2) {
  function create2(range2, parent2) {
    return { range: range2, parent: parent2 };
  }
  SelectionRange2.create = create2;
  function is(value) {
    let candidate = value;
    return Is.objectLiteral(candidate) && Range.is(candidate.range) && (candidate.parent === void 0 || SelectionRange2.is(candidate.parent));
  }
  SelectionRange2.is = is;
})(SelectionRange || (SelectionRange = {}));
var SemanticTokenTypes;
(function(SemanticTokenTypes2) {
  SemanticTokenTypes2["namespace"] = "namespace";
  SemanticTokenTypes2["type"] = "type";
  SemanticTokenTypes2["class"] = "class";
  SemanticTokenTypes2["enum"] = "enum";
  SemanticTokenTypes2["interface"] = "interface";
  SemanticTokenTypes2["struct"] = "struct";
  SemanticTokenTypes2["typeParameter"] = "typeParameter";
  SemanticTokenTypes2["parameter"] = "parameter";
  SemanticTokenTypes2["variable"] = "variable";
  SemanticTokenTypes2["property"] = "property";
  SemanticTokenTypes2["enumMember"] = "enumMember";
  SemanticTokenTypes2["event"] = "event";
  SemanticTokenTypes2["function"] = "function";
  SemanticTokenTypes2["method"] = "method";
  SemanticTokenTypes2["macro"] = "macro";
  SemanticTokenTypes2["keyword"] = "keyword";
  SemanticTokenTypes2["modifier"] = "modifier";
  SemanticTokenTypes2["comment"] = "comment";
  SemanticTokenTypes2["string"] = "string";
  SemanticTokenTypes2["number"] = "number";
  SemanticTokenTypes2["regexp"] = "regexp";
  SemanticTokenTypes2["operator"] = "operator";
  SemanticTokenTypes2["decorator"] = "decorator";
})(SemanticTokenTypes || (SemanticTokenTypes = {}));
var SemanticTokenModifiers;
(function(SemanticTokenModifiers2) {
  SemanticTokenModifiers2["declaration"] = "declaration";
  SemanticTokenModifiers2["definition"] = "definition";
  SemanticTokenModifiers2["readonly"] = "readonly";
  SemanticTokenModifiers2["static"] = "static";
  SemanticTokenModifiers2["deprecated"] = "deprecated";
  SemanticTokenModifiers2["abstract"] = "abstract";
  SemanticTokenModifiers2["async"] = "async";
  SemanticTokenModifiers2["modification"] = "modification";
  SemanticTokenModifiers2["documentation"] = "documentation";
  SemanticTokenModifiers2["defaultLibrary"] = "defaultLibrary";
})(SemanticTokenModifiers || (SemanticTokenModifiers = {}));
var SemanticTokens;
(function(SemanticTokens2) {
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && (candidate.resultId === void 0 || typeof candidate.resultId === "string") && Array.isArray(candidate.data) && (candidate.data.length === 0 || typeof candidate.data[0] === "number");
  }
  SemanticTokens2.is = is;
})(SemanticTokens || (SemanticTokens = {}));
var InlineValueText;
(function(InlineValueText2) {
  function create2(range2, text) {
    return { range: range2, text };
  }
  InlineValueText2.create = create2;
  function is(value) {
    const candidate = value;
    return candidate !== void 0 && candidate !== null && Range.is(candidate.range) && Is.string(candidate.text);
  }
  InlineValueText2.is = is;
})(InlineValueText || (InlineValueText = {}));
var InlineValueVariableLookup;
(function(InlineValueVariableLookup2) {
  function create2(range2, variableName, caseSensitiveLookup) {
    return { range: range2, variableName, caseSensitiveLookup };
  }
  InlineValueVariableLookup2.create = create2;
  function is(value) {
    const candidate = value;
    return candidate !== void 0 && candidate !== null && Range.is(candidate.range) && Is.boolean(candidate.caseSensitiveLookup) && (Is.string(candidate.variableName) || candidate.variableName === void 0);
  }
  InlineValueVariableLookup2.is = is;
})(InlineValueVariableLookup || (InlineValueVariableLookup = {}));
var InlineValueEvaluatableExpression;
(function(InlineValueEvaluatableExpression2) {
  function create2(range2, expression) {
    return { range: range2, expression };
  }
  InlineValueEvaluatableExpression2.create = create2;
  function is(value) {
    const candidate = value;
    return candidate !== void 0 && candidate !== null && Range.is(candidate.range) && (Is.string(candidate.expression) || candidate.expression === void 0);
  }
  InlineValueEvaluatableExpression2.is = is;
})(InlineValueEvaluatableExpression || (InlineValueEvaluatableExpression = {}));
var InlineValueContext;
(function(InlineValueContext2) {
  function create2(frameId, stoppedLocation) {
    return { frameId, stoppedLocation };
  }
  InlineValueContext2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.defined(candidate) && Range.is(value.stoppedLocation);
  }
  InlineValueContext2.is = is;
})(InlineValueContext || (InlineValueContext = {}));
var InlayHintKind;
(function(InlayHintKind2) {
  InlayHintKind2.Type = 1;
  InlayHintKind2.Parameter = 2;
  function is(value) {
    return value === 1 || value === 2;
  }
  InlayHintKind2.is = is;
})(InlayHintKind || (InlayHintKind = {}));
var InlayHintLabelPart;
(function(InlayHintLabelPart2) {
  function create2(value) {
    return { value };
  }
  InlayHintLabelPart2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && (candidate.tooltip === void 0 || Is.string(candidate.tooltip) || MarkupContent.is(candidate.tooltip)) && (candidate.location === void 0 || Location.is(candidate.location)) && (candidate.command === void 0 || Command.is(candidate.command));
  }
  InlayHintLabelPart2.is = is;
})(InlayHintLabelPart || (InlayHintLabelPart = {}));
var InlayHint;
(function(InlayHint2) {
  function create2(position, label, kind) {
    const result2 = { position, label };
    if (kind !== void 0) {
      result2.kind = kind;
    }
    return result2;
  }
  InlayHint2.create = create2;
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && Position.is(candidate.position) && (Is.string(candidate.label) || Is.typedArray(candidate.label, InlayHintLabelPart.is)) && (candidate.kind === void 0 || InlayHintKind.is(candidate.kind)) && candidate.textEdits === void 0 || Is.typedArray(candidate.textEdits, TextEdit.is) && (candidate.tooltip === void 0 || Is.string(candidate.tooltip) || MarkupContent.is(candidate.tooltip)) && (candidate.paddingLeft === void 0 || Is.boolean(candidate.paddingLeft)) && (candidate.paddingRight === void 0 || Is.boolean(candidate.paddingRight));
  }
  InlayHint2.is = is;
})(InlayHint || (InlayHint = {}));
var StringValue;
(function(StringValue2) {
  function createSnippet(value) {
    return { kind: "snippet", value };
  }
  StringValue2.createSnippet = createSnippet;
})(StringValue || (StringValue = {}));
var InlineCompletionItem;
(function(InlineCompletionItem2) {
  function create2(insertText, filterText, range2, command) {
    return { insertText, filterText, range: range2, command };
  }
  InlineCompletionItem2.create = create2;
})(InlineCompletionItem || (InlineCompletionItem = {}));
var InlineCompletionList;
(function(InlineCompletionList2) {
  function create2(items) {
    return { items };
  }
  InlineCompletionList2.create = create2;
})(InlineCompletionList || (InlineCompletionList = {}));
var InlineCompletionTriggerKind;
(function(InlineCompletionTriggerKind2) {
  InlineCompletionTriggerKind2.Invoked = 0;
  InlineCompletionTriggerKind2.Automatic = 1;
})(InlineCompletionTriggerKind || (InlineCompletionTriggerKind = {}));
var SelectedCompletionInfo;
(function(SelectedCompletionInfo2) {
  function create2(range2, text) {
    return { range: range2, text };
  }
  SelectedCompletionInfo2.create = create2;
})(SelectedCompletionInfo || (SelectedCompletionInfo = {}));
var InlineCompletionContext;
(function(InlineCompletionContext2) {
  function create2(triggerKind, selectedCompletionInfo) {
    return { triggerKind, selectedCompletionInfo };
  }
  InlineCompletionContext2.create = create2;
})(InlineCompletionContext || (InlineCompletionContext = {}));
var WorkspaceFolder;
(function(WorkspaceFolder2) {
  function is(value) {
    const candidate = value;
    return Is.objectLiteral(candidate) && URI.is(candidate.uri) && Is.string(candidate.name);
  }
  WorkspaceFolder2.is = is;
})(WorkspaceFolder || (WorkspaceFolder = {}));
var TextDocument;
(function(TextDocument3) {
  function create2(uri, languageId, version, content) {
    return new FullTextDocument(uri, languageId, version, content);
  }
  TextDocument3.create = create2;
  function is(value) {
    let candidate = value;
    return Is.defined(candidate) && Is.string(candidate.uri) && (Is.undefined(candidate.languageId) || Is.string(candidate.languageId)) && Is.uinteger(candidate.lineCount) && Is.func(candidate.getText) && Is.func(candidate.positionAt) && Is.func(candidate.offsetAt) ? true : false;
  }
  TextDocument3.is = is;
  function applyEdits(document, edits) {
    let text = document.getText();
    let sortedEdits = mergeSort2(edits, (a, b) => {
      let diff = a.range.start.line - b.range.start.line;
      if (diff === 0) {
        return a.range.start.character - b.range.start.character;
      }
      return diff;
    });
    let lastModifiedOffset = text.length;
    for (let i = sortedEdits.length - 1; i >= 0; i--) {
      let e = sortedEdits[i];
      let startOffset = document.offsetAt(e.range.start);
      let endOffset = document.offsetAt(e.range.end);
      if (endOffset <= lastModifiedOffset) {
        text = text.substring(0, startOffset) + e.newText + text.substring(endOffset, text.length);
      } else {
        throw new Error("Overlapping edit");
      }
      lastModifiedOffset = startOffset;
    }
    return text;
  }
  TextDocument3.applyEdits = applyEdits;
  function mergeSort2(data, compare) {
    if (data.length <= 1) {
      return data;
    }
    const p = data.length / 2 | 0;
    const left = data.slice(0, p);
    const right = data.slice(p);
    mergeSort2(left, compare);
    mergeSort2(right, compare);
    let leftIdx = 0;
    let rightIdx = 0;
    let i = 0;
    while (leftIdx < left.length && rightIdx < right.length) {
      let ret = compare(left[leftIdx], right[rightIdx]);
      if (ret <= 0) {
        data[i++] = left[leftIdx++];
      } else {
        data[i++] = right[rightIdx++];
      }
    }
    while (leftIdx < left.length) {
      data[i++] = left[leftIdx++];
    }
    while (rightIdx < right.length) {
      data[i++] = right[rightIdx++];
    }
    return data;
  }
})(TextDocument || (TextDocument = {}));
var FullTextDocument = class {
  constructor(uri, languageId, version, content) {
    this._uri = uri;
    this._languageId = languageId;
    this._version = version;
    this._content = content;
    this._lineOffsets = void 0;
  }
  get uri() {
    return this._uri;
  }
  get languageId() {
    return this._languageId;
  }
  get version() {
    return this._version;
  }
  getText(range2) {
    if (range2) {
      let start = this.offsetAt(range2.start);
      let end = this.offsetAt(range2.end);
      return this._content.substring(start, end);
    }
    return this._content;
  }
  update(event, version) {
    this._content = event.text;
    this._version = version;
    this._lineOffsets = void 0;
  }
  getLineOffsets() {
    if (this._lineOffsets === void 0) {
      let lineOffsets = [];
      let text = this._content;
      let isLineStart = true;
      for (let i = 0; i < text.length; i++) {
        if (isLineStart) {
          lineOffsets.push(i);
          isLineStart = false;
        }
        let ch = text.charAt(i);
        isLineStart = ch === "\r" || ch === "\n";
        if (ch === "\r" && i + 1 < text.length && text.charAt(i + 1) === "\n") {
          i++;
        }
      }
      if (isLineStart && text.length > 0) {
        lineOffsets.push(text.length);
      }
      this._lineOffsets = lineOffsets;
    }
    return this._lineOffsets;
  }
  positionAt(offset) {
    offset = Math.max(Math.min(offset, this._content.length), 0);
    let lineOffsets = this.getLineOffsets();
    let low = 0, high = lineOffsets.length;
    if (high === 0) {
      return Position.create(0, offset);
    }
    while (low < high) {
      let mid = Math.floor((low + high) / 2);
      if (lineOffsets[mid] > offset) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    let line = low - 1;
    return Position.create(line, offset - lineOffsets[line]);
  }
  offsetAt(position) {
    let lineOffsets = this.getLineOffsets();
    if (position.line >= lineOffsets.length) {
      return this._content.length;
    } else if (position.line < 0) {
      return 0;
    }
    let lineOffset = lineOffsets[position.line];
    let nextLineOffset = position.line + 1 < lineOffsets.length ? lineOffsets[position.line + 1] : this._content.length;
    return Math.max(Math.min(lineOffset + position.character, nextLineOffset), lineOffset);
  }
  get lineCount() {
    return this.getLineOffsets().length;
  }
};
var Is;
(function(Is2) {
  const toString3 = Object.prototype.toString;
  function defined(value) {
    return typeof value !== "undefined";
  }
  Is2.defined = defined;
  function undefined2(value) {
    return typeof value === "undefined";
  }
  Is2.undefined = undefined2;
  function boolean(value) {
    return value === true || value === false;
  }
  Is2.boolean = boolean;
  function string(value) {
    return toString3.call(value) === "[object String]";
  }
  Is2.string = string;
  function number(value) {
    return toString3.call(value) === "[object Number]";
  }
  Is2.number = number;
  function numberRange(value, min2, max2) {
    return toString3.call(value) === "[object Number]" && min2 <= value && value <= max2;
  }
  Is2.numberRange = numberRange;
  function integer2(value) {
    return toString3.call(value) === "[object Number]" && -2147483648 <= value && value <= 2147483647;
  }
  Is2.integer = integer2;
  function uinteger2(value) {
    return toString3.call(value) === "[object Number]" && 0 <= value && value <= 2147483647;
  }
  Is2.uinteger = uinteger2;
  function func(value) {
    return toString3.call(value) === "[object Function]";
  }
  Is2.func = func;
  function objectLiteral(value) {
    return value !== null && typeof value === "object";
  }
  Is2.objectLiteral = objectLiteral;
  function typedArray(value, check) {
    return Array.isArray(value) && value.every(check);
  }
  Is2.typedArray = typedArray;
})(Is || (Is = {}));

// node_modules/langium/lib/parser/cst-node-builder.js
var CstNodeBuilder = class {
  constructor() {
    this.nodeStack = [];
  }
  get current() {
    var _a;
    return (_a = this.nodeStack[this.nodeStack.length - 1]) !== null && _a !== void 0 ? _a : this.rootNode;
  }
  buildRootNode(input) {
    this.rootNode = new RootCstNodeImpl(input);
    this.rootNode.root = this.rootNode;
    this.nodeStack = [this.rootNode];
    return this.rootNode;
  }
  buildCompositeNode(feature) {
    const compositeNode = new CompositeCstNodeImpl();
    compositeNode.grammarSource = feature;
    compositeNode.root = this.rootNode;
    this.current.content.push(compositeNode);
    this.nodeStack.push(compositeNode);
    return compositeNode;
  }
  buildLeafNode(token, feature) {
    const leafNode = new LeafCstNodeImpl(token.startOffset, token.image.length, tokenToRange(token), token.tokenType, !feature);
    leafNode.grammarSource = feature;
    leafNode.root = this.rootNode;
    this.current.content.push(leafNode);
    return leafNode;
  }
  removeNode(node) {
    const parent2 = node.container;
    if (parent2) {
      const index = parent2.content.indexOf(node);
      if (index >= 0) {
        parent2.content.splice(index, 1);
      }
    }
  }
  addHiddenNodes(tokens) {
    const nodes = [];
    for (const token of tokens) {
      const leafNode = new LeafCstNodeImpl(token.startOffset, token.image.length, tokenToRange(token), token.tokenType, true);
      leafNode.root = this.rootNode;
      nodes.push(leafNode);
    }
    let current = this.current;
    let added = false;
    if (current.content.length > 0) {
      current.content.push(...nodes);
      return;
    }
    while (current.container) {
      const index = current.container.content.indexOf(current);
      if (index > 0) {
        current.container.content.splice(index, 0, ...nodes);
        added = true;
        break;
      }
      current = current.container;
    }
    if (!added) {
      this.rootNode.content.unshift(...nodes);
    }
  }
  construct(item) {
    const current = this.current;
    if (typeof item.$type === "string") {
      this.current.astNode = item;
    }
    item.$cstNode = current;
    const node = this.nodeStack.pop();
    if ((node === null || node === void 0 ? void 0 : node.content.length) === 0) {
      this.removeNode(node);
    }
  }
};
var AbstractCstNode = class {
  /** @deprecated use `container` instead. */
  get parent() {
    return this.container;
  }
  /** @deprecated use `grammarSource` instead. */
  get feature() {
    return this.grammarSource;
  }
  get hidden() {
    return false;
  }
  get astNode() {
    var _a, _b;
    const node = typeof ((_a = this._astNode) === null || _a === void 0 ? void 0 : _a.$type) === "string" ? this._astNode : (_b = this.container) === null || _b === void 0 ? void 0 : _b.astNode;
    if (!node) {
      throw new Error("This node has no associated AST element");
    }
    return node;
  }
  set astNode(value) {
    this._astNode = value;
  }
  /** @deprecated use `astNode` instead. */
  get element() {
    return this.astNode;
  }
  get text() {
    return this.root.fullText.substring(this.offset, this.end);
  }
};
var LeafCstNodeImpl = class extends AbstractCstNode {
  get offset() {
    return this._offset;
  }
  get length() {
    return this._length;
  }
  get end() {
    return this._offset + this._length;
  }
  get hidden() {
    return this._hidden;
  }
  get tokenType() {
    return this._tokenType;
  }
  get range() {
    return this._range;
  }
  constructor(offset, length, range2, tokenType, hidden = false) {
    super();
    this._hidden = hidden;
    this._offset = offset;
    this._tokenType = tokenType;
    this._length = length;
    this._range = range2;
  }
};
var CompositeCstNodeImpl = class extends AbstractCstNode {
  constructor() {
    super(...arguments);
    this.content = new CstNodeContainer(this);
  }
  /** @deprecated use `content` instead. */
  get children() {
    return this.content;
  }
  get offset() {
    var _a, _b;
    return (_b = (_a = this.firstNonHiddenNode) === null || _a === void 0 ? void 0 : _a.offset) !== null && _b !== void 0 ? _b : 0;
  }
  get length() {
    return this.end - this.offset;
  }
  get end() {
    var _a, _b;
    return (_b = (_a = this.lastNonHiddenNode) === null || _a === void 0 ? void 0 : _a.end) !== null && _b !== void 0 ? _b : 0;
  }
  get range() {
    const firstNode = this.firstNonHiddenNode;
    const lastNode = this.lastNonHiddenNode;
    if (firstNode && lastNode) {
      if (this._rangeCache === void 0) {
        const { range: firstRange } = firstNode;
        const { range: lastRange } = lastNode;
        this._rangeCache = { start: firstRange.start, end: lastRange.end.line < firstRange.start.line ? firstRange.start : lastRange.end };
      }
      return this._rangeCache;
    } else {
      return { start: Position.create(0, 0), end: Position.create(0, 0) };
    }
  }
  get firstNonHiddenNode() {
    for (const child of this.content) {
      if (!child.hidden) {
        return child;
      }
    }
    return this.content[0];
  }
  get lastNonHiddenNode() {
    for (let i = this.content.length - 1; i >= 0; i--) {
      const child = this.content[i];
      if (!child.hidden) {
        return child;
      }
    }
    return this.content[this.content.length - 1];
  }
};
var CstNodeContainer = class _CstNodeContainer extends Array {
  constructor(parent2) {
    super();
    this.parent = parent2;
    Object.setPrototypeOf(this, _CstNodeContainer.prototype);
  }
  push(...items) {
    this.addParents(items);
    return super.push(...items);
  }
  unshift(...items) {
    this.addParents(items);
    return super.unshift(...items);
  }
  splice(start, count, ...items) {
    this.addParents(items);
    return super.splice(start, count, ...items);
  }
  addParents(items) {
    for (const item of items) {
      item.container = this.parent;
    }
  }
};
var RootCstNodeImpl = class extends CompositeCstNodeImpl {
  get text() {
    return this._text.substring(this.offset, this.end);
  }
  get fullText() {
    return this._text;
  }
  constructor(input) {
    super();
    this._text = "";
    this._text = input !== null && input !== void 0 ? input : "";
  }
};

// node_modules/langium/lib/parser/langium-parser.js
var DatatypeSymbol = Symbol("Datatype");
function isDataTypeNode(node) {
  return node.$type === DatatypeSymbol;
}
var ruleSuffix = "​";
var withRuleSuffix = (name) => name.endsWith(ruleSuffix) ? name : name + ruleSuffix;
var AbstractLangiumParser = class {
  constructor(services) {
    this._unorderedGroups = /* @__PURE__ */ new Map();
    this.allRules = /* @__PURE__ */ new Map();
    this.lexer = services.parser.Lexer;
    const tokens = this.lexer.definition;
    const production = services.LanguageMetaData.mode === "production";
    this.wrapper = new ChevrotainWrapper(tokens, Object.assign(Object.assign({}, services.parser.ParserConfig), { skipValidations: production, errorMessageProvider: services.parser.ParserErrorMessageProvider }));
  }
  alternatives(idx, choices) {
    this.wrapper.wrapOr(idx, choices);
  }
  optional(idx, callback) {
    this.wrapper.wrapOption(idx, callback);
  }
  many(idx, callback) {
    this.wrapper.wrapMany(idx, callback);
  }
  atLeastOne(idx, callback) {
    this.wrapper.wrapAtLeastOne(idx, callback);
  }
  getRule(name) {
    return this.allRules.get(name);
  }
  isRecording() {
    return this.wrapper.IS_RECORDING;
  }
  get unorderedGroups() {
    return this._unorderedGroups;
  }
  getRuleStack() {
    return this.wrapper.RULE_STACK;
  }
  finalize() {
    this.wrapper.wrapSelfAnalysis();
  }
};
var LangiumParser = class extends AbstractLangiumParser {
  get current() {
    return this.stack[this.stack.length - 1];
  }
  constructor(services) {
    super(services);
    this.nodeBuilder = new CstNodeBuilder();
    this.stack = [];
    this.assignmentMap = /* @__PURE__ */ new Map();
    this.linker = services.references.Linker;
    this.converter = services.parser.ValueConverter;
    this.astReflection = services.shared.AstReflection;
  }
  rule(rule, impl) {
    const type = this.computeRuleType(rule);
    const ruleMethod = this.wrapper.DEFINE_RULE(withRuleSuffix(rule.name), this.startImplementation(type, impl).bind(this));
    this.allRules.set(rule.name, ruleMethod);
    if (rule.entry) {
      this.mainRule = ruleMethod;
    }
    return ruleMethod;
  }
  computeRuleType(rule) {
    if (rule.fragment) {
      return void 0;
    } else if (isDataTypeRule(rule)) {
      return DatatypeSymbol;
    } else {
      const explicit = getExplicitRuleType(rule);
      return explicit !== null && explicit !== void 0 ? explicit : rule.name;
    }
  }
  parse(input, options = {}) {
    this.nodeBuilder.buildRootNode(input);
    const lexerResult = this.lexerResult = this.lexer.tokenize(input);
    this.wrapper.input = lexerResult.tokens;
    const ruleMethod = options.rule ? this.allRules.get(options.rule) : this.mainRule;
    if (!ruleMethod) {
      throw new Error(options.rule ? `No rule found with name '${options.rule}'` : "No main rule available.");
    }
    const result2 = ruleMethod.call(this.wrapper, {});
    this.nodeBuilder.addHiddenNodes(lexerResult.hidden);
    this.unorderedGroups.clear();
    this.lexerResult = void 0;
    return {
      value: result2,
      lexerErrors: lexerResult.errors,
      lexerReport: lexerResult.report,
      parserErrors: this.wrapper.errors
    };
  }
  startImplementation($type, implementation) {
    return (args) => {
      const createNode = !this.isRecording() && $type !== void 0;
      if (createNode) {
        const node = { $type };
        this.stack.push(node);
        if ($type === DatatypeSymbol) {
          node.value = "";
        }
      }
      let result2;
      try {
        result2 = implementation(args);
      } catch (err) {
        result2 = void 0;
      }
      if (result2 === void 0 && createNode) {
        result2 = this.construct();
      }
      return result2;
    };
  }
  extractHiddenTokens(token) {
    const hiddenTokens = this.lexerResult.hidden;
    if (!hiddenTokens.length) {
      return [];
    }
    const offset = token.startOffset;
    for (let i = 0; i < hiddenTokens.length; i++) {
      const token2 = hiddenTokens[i];
      if (token2.startOffset > offset) {
        return hiddenTokens.splice(0, i);
      }
    }
    return hiddenTokens.splice(0, hiddenTokens.length);
  }
  consume(idx, tokenType, feature) {
    const token = this.wrapper.wrapConsume(idx, tokenType);
    if (!this.isRecording() && this.isValidToken(token)) {
      const hiddenTokens = this.extractHiddenTokens(token);
      this.nodeBuilder.addHiddenNodes(hiddenTokens);
      const leafNode = this.nodeBuilder.buildLeafNode(token, feature);
      const { assignment, isCrossRef } = this.getAssignment(feature);
      const current = this.current;
      if (assignment) {
        const convertedValue = isKeyword(feature) ? token.image : this.converter.convert(token.image, leafNode);
        this.assign(assignment.operator, assignment.feature, convertedValue, leafNode, isCrossRef);
      } else if (isDataTypeNode(current)) {
        let text = token.image;
        if (!isKeyword(feature)) {
          text = this.converter.convert(text, leafNode).toString();
        }
        current.value += text;
      }
    }
  }
  /**
   * Most consumed parser tokens are valid. However there are two cases in which they are not valid:
   *
   * 1. They were inserted during error recovery by the parser. These tokens don't really exist and should not be further processed
   * 2. They contain invalid token ranges. This might include the special EOF token, or other tokens produced by invalid token builders.
   */
  isValidToken(token) {
    return !token.isInsertedInRecovery && !isNaN(token.startOffset) && typeof token.endOffset === "number" && !isNaN(token.endOffset);
  }
  subrule(idx, rule, fragment, feature, args) {
    let cstNode;
    if (!this.isRecording() && !fragment) {
      cstNode = this.nodeBuilder.buildCompositeNode(feature);
    }
    const subruleResult = this.wrapper.wrapSubrule(idx, rule, args);
    if (!this.isRecording() && cstNode && cstNode.length > 0) {
      this.performSubruleAssignment(subruleResult, feature, cstNode);
    }
  }
  performSubruleAssignment(result2, feature, cstNode) {
    const { assignment, isCrossRef } = this.getAssignment(feature);
    if (assignment) {
      this.assign(assignment.operator, assignment.feature, result2, cstNode, isCrossRef);
    } else if (!assignment) {
      const current = this.current;
      if (isDataTypeNode(current)) {
        current.value += result2.toString();
      } else if (typeof result2 === "object" && result2) {
        const object = this.assignWithoutOverride(result2, current);
        const newItem = object;
        this.stack.pop();
        this.stack.push(newItem);
      }
    }
  }
  action($type, action) {
    if (!this.isRecording()) {
      let last2 = this.current;
      if (action.feature && action.operator) {
        last2 = this.construct();
        this.nodeBuilder.removeNode(last2.$cstNode);
        const node = this.nodeBuilder.buildCompositeNode(action);
        node.content.push(last2.$cstNode);
        const newItem = { $type };
        this.stack.push(newItem);
        this.assign(action.operator, action.feature, last2, last2.$cstNode, false);
      } else {
        last2.$type = $type;
      }
    }
  }
  construct() {
    if (this.isRecording()) {
      return void 0;
    }
    const obj = this.current;
    linkContentToContainer(obj);
    this.nodeBuilder.construct(obj);
    this.stack.pop();
    if (isDataTypeNode(obj)) {
      return this.converter.convert(obj.value, obj.$cstNode);
    } else {
      assignMandatoryProperties(this.astReflection, obj);
    }
    return obj;
  }
  getAssignment(feature) {
    if (!this.assignmentMap.has(feature)) {
      const assignment = getContainerOfType(feature, isAssignment);
      this.assignmentMap.set(feature, {
        assignment,
        isCrossRef: assignment ? isCrossReference(assignment.terminal) : false
      });
    }
    return this.assignmentMap.get(feature);
  }
  assign(operator, feature, value, cstNode, isCrossRef) {
    const obj = this.current;
    let item;
    if (isCrossRef && typeof value === "string") {
      item = this.linker.buildReference(obj, feature, cstNode, value);
    } else {
      item = value;
    }
    switch (operator) {
      case "=": {
        obj[feature] = item;
        break;
      }
      case "?=": {
        obj[feature] = true;
        break;
      }
      case "+=": {
        if (!Array.isArray(obj[feature])) {
          obj[feature] = [];
        }
        obj[feature].push(item);
      }
    }
  }
  assignWithoutOverride(target, source) {
    for (const [name, existingValue] of Object.entries(source)) {
      const newValue = target[name];
      if (newValue === void 0) {
        target[name] = existingValue;
      } else if (Array.isArray(newValue) && Array.isArray(existingValue)) {
        existingValue.push(...newValue);
        target[name] = existingValue;
      }
    }
    const targetCstNode = target.$cstNode;
    if (targetCstNode) {
      targetCstNode.astNode = void 0;
      target.$cstNode = void 0;
    }
    return target;
  }
  get definitionErrors() {
    return this.wrapper.definitionErrors;
  }
};
var AbstractParserErrorMessageProvider = class {
  buildMismatchTokenMessage(options) {
    return defaultParserErrorProvider.buildMismatchTokenMessage(options);
  }
  buildNotAllInputParsedMessage(options) {
    return defaultParserErrorProvider.buildNotAllInputParsedMessage(options);
  }
  buildNoViableAltMessage(options) {
    return defaultParserErrorProvider.buildNoViableAltMessage(options);
  }
  buildEarlyExitMessage(options) {
    return defaultParserErrorProvider.buildEarlyExitMessage(options);
  }
};
var LangiumParserErrorMessageProvider = class extends AbstractParserErrorMessageProvider {
  buildMismatchTokenMessage({ expected, actual }) {
    const expectedMsg = expected.LABEL ? "`" + expected.LABEL + "`" : expected.name.endsWith(":KW") ? `keyword '${expected.name.substring(0, expected.name.length - 3)}'` : `token of type '${expected.name}'`;
    return `Expecting ${expectedMsg} but found \`${actual.image}\`.`;
  }
  buildNotAllInputParsedMessage({ firstRedundant }) {
    return `Expecting end of file but found \`${firstRedundant.image}\`.`;
  }
};
var LangiumCompletionParser = class extends AbstractLangiumParser {
  constructor() {
    super(...arguments);
    this.tokens = [];
    this.elementStack = [];
    this.lastElementStack = [];
    this.nextTokenIndex = 0;
    this.stackSize = 0;
  }
  action() {
  }
  construct() {
    return void 0;
  }
  parse(input) {
    this.resetState();
    const tokens = this.lexer.tokenize(input, { mode: "partial" });
    this.tokens = tokens.tokens;
    this.wrapper.input = [...this.tokens];
    this.mainRule.call(this.wrapper, {});
    this.unorderedGroups.clear();
    return {
      tokens: this.tokens,
      elementStack: [...this.lastElementStack],
      tokenIndex: this.nextTokenIndex
    };
  }
  rule(rule, impl) {
    const ruleMethod = this.wrapper.DEFINE_RULE(withRuleSuffix(rule.name), this.startImplementation(impl).bind(this));
    this.allRules.set(rule.name, ruleMethod);
    if (rule.entry) {
      this.mainRule = ruleMethod;
    }
    return ruleMethod;
  }
  resetState() {
    this.elementStack = [];
    this.lastElementStack = [];
    this.nextTokenIndex = 0;
    this.stackSize = 0;
  }
  startImplementation(implementation) {
    return (args) => {
      const size2 = this.keepStackSize();
      try {
        implementation(args);
      } finally {
        this.resetStackSize(size2);
      }
    };
  }
  removeUnexpectedElements() {
    this.elementStack.splice(this.stackSize);
  }
  keepStackSize() {
    const size2 = this.elementStack.length;
    this.stackSize = size2;
    return size2;
  }
  resetStackSize(size2) {
    this.removeUnexpectedElements();
    this.stackSize = size2;
  }
  consume(idx, tokenType, feature) {
    this.wrapper.wrapConsume(idx, tokenType);
    if (!this.isRecording()) {
      this.lastElementStack = [...this.elementStack, feature];
      this.nextTokenIndex = this.currIdx + 1;
    }
  }
  subrule(idx, rule, fragment, feature, args) {
    this.before(feature);
    this.wrapper.wrapSubrule(idx, rule, args);
    this.after(feature);
  }
  before(element) {
    if (!this.isRecording()) {
      this.elementStack.push(element);
    }
  }
  after(element) {
    if (!this.isRecording()) {
      const index = this.elementStack.lastIndexOf(element);
      if (index >= 0) {
        this.elementStack.splice(index);
      }
    }
  }
  get currIdx() {
    return this.wrapper.currIdx;
  }
};
var defaultConfig = {
  recoveryEnabled: true,
  nodeLocationTracking: "full",
  skipValidations: true,
  errorMessageProvider: new LangiumParserErrorMessageProvider()
};
var ChevrotainWrapper = class extends EmbeddedActionsParser {
  constructor(tokens, config) {
    const useDefaultLookahead = config && "maxLookahead" in config;
    super(tokens, Object.assign(Object.assign(Object.assign({}, defaultConfig), { lookaheadStrategy: useDefaultLookahead ? new LLkLookaheadStrategy({ maxLookahead: config.maxLookahead }) : new LLStarLookaheadStrategy({
      // If validations are skipped, don't log the lookahead warnings
      logging: config.skipValidations ? () => {
      } : void 0
    }) }), config));
  }
  get IS_RECORDING() {
    return this.RECORDING_PHASE;
  }
  DEFINE_RULE(name, impl) {
    return this.RULE(name, impl);
  }
  wrapSelfAnalysis() {
    this.performSelfAnalysis();
  }
  wrapConsume(idx, tokenType) {
    return this.consume(idx, tokenType);
  }
  wrapSubrule(idx, rule, args) {
    return this.subrule(idx, rule, {
      ARGS: [args]
    });
  }
  wrapOr(idx, choices) {
    this.or(idx, choices);
  }
  wrapOption(idx, callback) {
    this.option(idx, callback);
  }
  wrapMany(idx, callback) {
    this.many(idx, callback);
  }
  wrapAtLeastOne(idx, callback) {
    this.atLeastOne(idx, callback);
  }
};

// node_modules/langium/lib/parser/parser-builder-base.js
function createParser(grammar, parser, tokens) {
  const parserContext = {
    parser,
    tokens,
    ruleNames: /* @__PURE__ */ new Map()
  };
  buildRules(parserContext, grammar);
  return parser;
}
function buildRules(parserContext, grammar) {
  const reachable = getAllReachableRules(grammar, false);
  const parserRules = stream(grammar.rules).filter(isParserRule).filter((rule) => reachable.has(rule));
  for (const rule of parserRules) {
    const ctx = Object.assign(Object.assign({}, parserContext), { consume: 1, optional: 1, subrule: 1, many: 1, or: 1 });
    parserContext.parser.rule(rule, buildElement(ctx, rule.definition));
  }
}
function buildElement(ctx, element, ignoreGuard = false) {
  let method2;
  if (isKeyword(element)) {
    method2 = buildKeyword(ctx, element);
  } else if (isAction(element)) {
    method2 = buildAction(ctx, element);
  } else if (isAssignment(element)) {
    method2 = buildElement(ctx, element.terminal);
  } else if (isCrossReference(element)) {
    method2 = buildCrossReference(ctx, element);
  } else if (isRuleCall(element)) {
    method2 = buildRuleCall(ctx, element);
  } else if (isAlternatives(element)) {
    method2 = buildAlternatives(ctx, element);
  } else if (isUnorderedGroup(element)) {
    method2 = buildUnorderedGroup(ctx, element);
  } else if (isGroup(element)) {
    method2 = buildGroup(ctx, element);
  } else if (isEndOfFile(element)) {
    const idx = ctx.consume++;
    method2 = () => ctx.parser.consume(idx, EOF, element);
  } else {
    throw new ErrorWithLocation(element.$cstNode, `Unexpected element type: ${element.$type}`);
  }
  return wrap2(ctx, ignoreGuard ? void 0 : getGuardCondition(element), method2, element.cardinality);
}
function buildAction(ctx, action) {
  const actionType = getTypeName(action);
  return () => ctx.parser.action(actionType, action);
}
function buildRuleCall(ctx, ruleCall) {
  const rule = ruleCall.rule.ref;
  if (isParserRule(rule)) {
    const idx = ctx.subrule++;
    const fragment = rule.fragment;
    const predicate = ruleCall.arguments.length > 0 ? buildRuleCallPredicate(rule, ruleCall.arguments) : () => ({});
    return (args) => ctx.parser.subrule(idx, getRule(ctx, rule), fragment, ruleCall, predicate(args));
  } else if (isTerminalRule(rule)) {
    const idx = ctx.consume++;
    const method2 = getToken(ctx, rule.name);
    return () => ctx.parser.consume(idx, method2, ruleCall);
  } else if (!rule) {
    throw new ErrorWithLocation(ruleCall.$cstNode, `Undefined rule: ${ruleCall.rule.$refText}`);
  } else {
    assertUnreachable(rule);
  }
}
function buildRuleCallPredicate(rule, namedArgs) {
  const predicates = namedArgs.map((e) => buildPredicate(e.value));
  return (args) => {
    const ruleArgs = {};
    for (let i = 0; i < predicates.length; i++) {
      const ruleTarget = rule.parameters[i];
      const predicate = predicates[i];
      ruleArgs[ruleTarget.name] = predicate(args);
    }
    return ruleArgs;
  };
}
function buildPredicate(condition) {
  if (isDisjunction(condition)) {
    const left = buildPredicate(condition.left);
    const right = buildPredicate(condition.right);
    return (args) => left(args) || right(args);
  } else if (isConjunction(condition)) {
    const left = buildPredicate(condition.left);
    const right = buildPredicate(condition.right);
    return (args) => left(args) && right(args);
  } else if (isNegation(condition)) {
    const value = buildPredicate(condition.value);
    return (args) => !value(args);
  } else if (isParameterReference(condition)) {
    const name = condition.parameter.ref.name;
    return (args) => args !== void 0 && args[name] === true;
  } else if (isBooleanLiteral(condition)) {
    const value = Boolean(condition.true);
    return () => value;
  }
  assertUnreachable(condition);
}
function buildAlternatives(ctx, alternatives) {
  if (alternatives.elements.length === 1) {
    return buildElement(ctx, alternatives.elements[0]);
  } else {
    const methods = [];
    for (const element of alternatives.elements) {
      const predicatedMethod = {
        // Since we handle the guard condition in the alternative already
        // We can ignore the group guard condition inside
        ALT: buildElement(ctx, element, true)
      };
      const guard = getGuardCondition(element);
      if (guard) {
        predicatedMethod.GATE = buildPredicate(guard);
      }
      methods.push(predicatedMethod);
    }
    const idx = ctx.or++;
    return (args) => ctx.parser.alternatives(idx, methods.map((method2) => {
      const alt = {
        ALT: () => method2.ALT(args)
      };
      const gate = method2.GATE;
      if (gate) {
        alt.GATE = () => gate(args);
      }
      return alt;
    }));
  }
}
function buildUnorderedGroup(ctx, group) {
  if (group.elements.length === 1) {
    return buildElement(ctx, group.elements[0]);
  }
  const methods = [];
  for (const element of group.elements) {
    const predicatedMethod = {
      // Since we handle the guard condition in the alternative already
      // We can ignore the group guard condition inside
      ALT: buildElement(ctx, element, true)
    };
    const guard = getGuardCondition(element);
    if (guard) {
      predicatedMethod.GATE = buildPredicate(guard);
    }
    methods.push(predicatedMethod);
  }
  const orIdx = ctx.or++;
  const idFunc = (groupIdx, lParser) => {
    const stackId = lParser.getRuleStack().join("-");
    return `uGroup_${groupIdx}_${stackId}`;
  };
  const alternatives = (args) => ctx.parser.alternatives(orIdx, methods.map((method2, idx) => {
    const alt = { ALT: () => true };
    const parser = ctx.parser;
    alt.ALT = () => {
      method2.ALT(args);
      if (!parser.isRecording()) {
        const key = idFunc(orIdx, parser);
        if (!parser.unorderedGroups.get(key)) {
          parser.unorderedGroups.set(key, []);
        }
        const groupState = parser.unorderedGroups.get(key);
        if (typeof (groupState === null || groupState === void 0 ? void 0 : groupState[idx]) === "undefined") {
          groupState[idx] = true;
        }
      }
    };
    const gate = method2.GATE;
    if (gate) {
      alt.GATE = () => gate(args);
    } else {
      alt.GATE = () => {
        const trackedAlternatives = parser.unorderedGroups.get(idFunc(orIdx, parser));
        const allow = !(trackedAlternatives === null || trackedAlternatives === void 0 ? void 0 : trackedAlternatives[idx]);
        return allow;
      };
    }
    return alt;
  }));
  const wrapped = wrap2(ctx, getGuardCondition(group), alternatives, "*");
  return (args) => {
    wrapped(args);
    if (!ctx.parser.isRecording()) {
      ctx.parser.unorderedGroups.delete(idFunc(orIdx, ctx.parser));
    }
  };
}
function buildGroup(ctx, group) {
  const methods = group.elements.map((e) => buildElement(ctx, e));
  return (args) => methods.forEach((method2) => method2(args));
}
function getGuardCondition(element) {
  if (isGroup(element)) {
    return element.guardCondition;
  }
  return void 0;
}
function buildCrossReference(ctx, crossRef, terminal = crossRef.terminal) {
  if (!terminal) {
    if (!crossRef.type.ref) {
      throw new Error("Could not resolve reference to type: " + crossRef.type.$refText);
    }
    const assignment = findNameAssignment(crossRef.type.ref);
    const assignTerminal = assignment === null || assignment === void 0 ? void 0 : assignment.terminal;
    if (!assignTerminal) {
      throw new Error("Could not find name assignment for type: " + getTypeName(crossRef.type.ref));
    }
    return buildCrossReference(ctx, crossRef, assignTerminal);
  } else if (isRuleCall(terminal) && isParserRule(terminal.rule.ref)) {
    const rule = terminal.rule.ref;
    const idx = ctx.subrule++;
    return (args) => ctx.parser.subrule(idx, getRule(ctx, rule), false, crossRef, args);
  } else if (isRuleCall(terminal) && isTerminalRule(terminal.rule.ref)) {
    const idx = ctx.consume++;
    const terminalRule = getToken(ctx, terminal.rule.ref.name);
    return () => ctx.parser.consume(idx, terminalRule, crossRef);
  } else if (isKeyword(terminal)) {
    const idx = ctx.consume++;
    const keyword = getToken(ctx, terminal.value);
    return () => ctx.parser.consume(idx, keyword, crossRef);
  } else {
    throw new Error("Could not build cross reference parser");
  }
}
function buildKeyword(ctx, keyword) {
  const idx = ctx.consume++;
  const token = ctx.tokens[keyword.value];
  if (!token) {
    throw new Error("Could not find token for keyword: " + keyword.value);
  }
  return () => ctx.parser.consume(idx, token, keyword);
}
function wrap2(ctx, guard, method2, cardinality) {
  const gate = guard && buildPredicate(guard);
  if (!cardinality) {
    if (gate) {
      const idx = ctx.or++;
      return (args) => ctx.parser.alternatives(idx, [
        {
          ALT: () => method2(args),
          GATE: () => gate(args)
        },
        {
          ALT: EMPTY_ALT(),
          GATE: () => !gate(args)
        }
      ]);
    } else {
      return method2;
    }
  }
  if (cardinality === "*") {
    const idx = ctx.many++;
    return (args) => ctx.parser.many(idx, {
      DEF: () => method2(args),
      GATE: gate ? () => gate(args) : void 0
    });
  } else if (cardinality === "+") {
    const idx = ctx.many++;
    if (gate) {
      const orIdx = ctx.or++;
      return (args) => ctx.parser.alternatives(orIdx, [
        {
          ALT: () => ctx.parser.atLeastOne(idx, {
            DEF: () => method2(args)
          }),
          GATE: () => gate(args)
        },
        {
          ALT: EMPTY_ALT(),
          GATE: () => !gate(args)
        }
      ]);
    } else {
      return (args) => ctx.parser.atLeastOne(idx, {
        DEF: () => method2(args)
      });
    }
  } else if (cardinality === "?") {
    const idx = ctx.optional++;
    return (args) => ctx.parser.optional(idx, {
      DEF: () => method2(args),
      GATE: gate ? () => gate(args) : void 0
    });
  } else {
    assertUnreachable(cardinality);
  }
}
function getRule(ctx, element) {
  const name = getRuleName(ctx, element);
  const rule = ctx.parser.getRule(name);
  if (!rule)
    throw new Error(`Rule "${name}" not found."`);
  return rule;
}
function getRuleName(ctx, element) {
  if (isParserRule(element)) {
    return element.name;
  } else if (ctx.ruleNames.has(element)) {
    return ctx.ruleNames.get(element);
  } else {
    let item = element;
    let parent2 = item.$container;
    let ruleName = element.$type;
    while (!isParserRule(parent2)) {
      if (isGroup(parent2) || isAlternatives(parent2) || isUnorderedGroup(parent2)) {
        const index = parent2.elements.indexOf(item);
        ruleName = index.toString() + ":" + ruleName;
      }
      item = parent2;
      parent2 = parent2.$container;
    }
    const rule = parent2;
    ruleName = rule.name + ":" + ruleName;
    ctx.ruleNames.set(element, ruleName);
    return ruleName;
  }
}
function getToken(ctx, name) {
  const token = ctx.tokens[name];
  if (!token)
    throw new Error(`Token "${name}" not found."`);
  return token;
}

// node_modules/langium/lib/parser/completion-parser-builder.js
function createCompletionParser(services) {
  const grammar = services.Grammar;
  const lexer = services.parser.Lexer;
  const parser = new LangiumCompletionParser(services);
  createParser(grammar, parser, lexer.definition);
  parser.finalize();
  return parser;
}

// node_modules/langium/lib/parser/langium-parser-builder.js
function createLangiumParser(services) {
  const parser = prepareLangiumParser(services);
  parser.finalize();
  return parser;
}
function prepareLangiumParser(services) {
  const grammar = services.Grammar;
  const lexer = services.parser.Lexer;
  const parser = new LangiumParser(services);
  return createParser(grammar, parser, lexer.definition);
}

// node_modules/langium/lib/parser/token-builder.js
var DefaultTokenBuilder = class {
  constructor() {
    this.diagnostics = [];
  }
  buildTokens(grammar, options) {
    const reachableRules = stream(getAllReachableRules(grammar, false));
    const terminalTokens = this.buildTerminalTokens(reachableRules);
    const tokens = this.buildKeywordTokens(reachableRules, terminalTokens, options);
    terminalTokens.forEach((terminalToken) => {
      const pattern = terminalToken.PATTERN;
      if (typeof pattern === "object" && pattern && "test" in pattern && isWhitespace(pattern)) {
        tokens.unshift(terminalToken);
      } else {
        tokens.push(terminalToken);
      }
    });
    return tokens;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  flushLexingReport(text) {
    return { diagnostics: this.popDiagnostics() };
  }
  popDiagnostics() {
    const diagnostics = [...this.diagnostics];
    this.diagnostics = [];
    return diagnostics;
  }
  buildTerminalTokens(rules) {
    return rules.filter(isTerminalRule).filter((e) => !e.fragment).map((terminal) => this.buildTerminalToken(terminal)).toArray();
  }
  buildTerminalToken(terminal) {
    const regex = terminalRegex(terminal);
    const pattern = this.requiresCustomPattern(regex) ? this.regexPatternFunction(regex) : regex;
    const tokenType = {
      name: terminal.name,
      PATTERN: pattern
    };
    if (typeof pattern === "function") {
      tokenType.LINE_BREAKS = true;
    }
    if (terminal.hidden) {
      tokenType.GROUP = isWhitespace(regex) ? Lexer.SKIPPED : "hidden";
    }
    return tokenType;
  }
  requiresCustomPattern(regex) {
    if (regex.flags.includes("u") || regex.flags.includes("s")) {
      return true;
    } else if (regex.source.includes("?<=") || regex.source.includes("?<!")) {
      return true;
    } else {
      return false;
    }
  }
  regexPatternFunction(regex) {
    const stickyRegex = new RegExp(regex, regex.flags + "y");
    return (text, offset) => {
      stickyRegex.lastIndex = offset;
      const execResult = stickyRegex.exec(text);
      return execResult;
    };
  }
  buildKeywordTokens(rules, terminalTokens, options) {
    return rules.filter(isParserRule).flatMap((rule) => streamAllContents(rule).filter(isKeyword)).distinct((e) => e.value).toArray().sort((a, b) => b.value.length - a.value.length).map((keyword) => this.buildKeywordToken(keyword, terminalTokens, Boolean(options === null || options === void 0 ? void 0 : options.caseInsensitive)));
  }
  buildKeywordToken(keyword, terminalTokens, caseInsensitive) {
    const keywordPattern = this.buildKeywordPattern(keyword, caseInsensitive);
    const tokenType = {
      name: keyword.value,
      PATTERN: keywordPattern,
      LONGER_ALT: this.findLongerAlt(keyword, terminalTokens)
    };
    if (typeof keywordPattern === "function") {
      tokenType.LINE_BREAKS = true;
    }
    return tokenType;
  }
  buildKeywordPattern(keyword, caseInsensitive) {
    return caseInsensitive ? new RegExp(getCaseInsensitivePattern(keyword.value)) : keyword.value;
  }
  findLongerAlt(keyword, terminalTokens) {
    return terminalTokens.reduce((longerAlts, token) => {
      const pattern = token === null || token === void 0 ? void 0 : token.PATTERN;
      if ((pattern === null || pattern === void 0 ? void 0 : pattern.source) && partialMatches("^" + pattern.source + "$", keyword.value)) {
        longerAlts.push(token);
      }
      return longerAlts;
    }, []);
  }
};

// node_modules/langium/lib/parser/value-converter.js
var DefaultValueConverter = class {
  convert(input, cstNode) {
    let feature = cstNode.grammarSource;
    if (isCrossReference(feature)) {
      feature = getCrossReferenceTerminal(feature);
    }
    if (isRuleCall(feature)) {
      const rule = feature.rule.ref;
      if (!rule) {
        throw new Error("This cst node was not parsed by a rule.");
      }
      return this.runConverter(rule, input, cstNode);
    }
    return input;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  runConverter(rule, input, cstNode) {
    var _a;
    switch (rule.name.toUpperCase()) {
      case "INT":
        return ValueConverter.convertInt(input);
      case "STRING":
        return ValueConverter.convertString(input);
      case "ID":
        return ValueConverter.convertID(input);
    }
    switch ((_a = getRuleType(rule)) === null || _a === void 0 ? void 0 : _a.toLowerCase()) {
      case "number":
        return ValueConverter.convertNumber(input);
      case "boolean":
        return ValueConverter.convertBoolean(input);
      case "bigint":
        return ValueConverter.convertBigint(input);
      case "date":
        return ValueConverter.convertDate(input);
      default:
        return input;
    }
  }
};
var ValueConverter;
(function(ValueConverter2) {
  function convertString(input) {
    let result2 = "";
    for (let i = 1; i < input.length - 1; i++) {
      const c = input.charAt(i);
      if (c === "\\") {
        const c1 = input.charAt(++i);
        result2 += convertEscapeCharacter(c1);
      } else {
        result2 += c;
      }
    }
    return result2;
  }
  ValueConverter2.convertString = convertString;
  function convertEscapeCharacter(char) {
    switch (char) {
      case "b":
        return "\b";
      case "f":
        return "\f";
      case "n":
        return "\n";
      case "r":
        return "\r";
      case "t":
        return "	";
      case "v":
        return "\v";
      case "0":
        return "\0";
      default:
        return char;
    }
  }
  function convertID(input) {
    if (input.charAt(0) === "^") {
      return input.substring(1);
    } else {
      return input;
    }
  }
  ValueConverter2.convertID = convertID;
  function convertInt(input) {
    return parseInt(input);
  }
  ValueConverter2.convertInt = convertInt;
  function convertBigint(input) {
    return BigInt(input);
  }
  ValueConverter2.convertBigint = convertBigint;
  function convertDate(input) {
    return new Date(input);
  }
  ValueConverter2.convertDate = convertDate;
  function convertNumber(input) {
    return Number(input);
  }
  ValueConverter2.convertNumber = convertNumber;
  function convertBoolean(input) {
    return input.toLowerCase() === "true";
  }
  ValueConverter2.convertBoolean = convertBoolean;
})(ValueConverter || (ValueConverter = {}));

// node_modules/langium/lib/utils/cancellation.js
var cancellation_exports = {};
__reExport(cancellation_exports, __toESM(require_cancellation(), 1));

// node_modules/langium/lib/utils/promise-utils.js
function delayNextTick() {
  return new Promise((resolve) => {
    if (typeof setImmediate === "undefined") {
      setTimeout(resolve, 0);
    } else {
      setImmediate(resolve);
    }
  });
}
var lastTick = 0;
var globalInterruptionPeriod = 10;
function startCancelableOperation() {
  lastTick = performance.now();
  return new cancellation_exports.CancellationTokenSource();
}
function setInterruptionPeriod(period) {
  globalInterruptionPeriod = period;
}
var OperationCancelled = Symbol("OperationCancelled");
function isOperationCancelled(err) {
  return err === OperationCancelled;
}
async function interruptAndCheck(token) {
  if (token === cancellation_exports.CancellationToken.None) {
    return;
  }
  const current = performance.now();
  if (current - lastTick >= globalInterruptionPeriod) {
    lastTick = current;
    await delayNextTick();
    lastTick = performance.now();
  }
  if (token.isCancellationRequested) {
    throw OperationCancelled;
  }
}
var Deferred = class {
  constructor() {
    this.promise = new Promise((resolve, reject2) => {
      this.resolve = (arg) => {
        resolve(arg);
        return this;
      };
      this.reject = (err) => {
        reject2(err);
        return this;
      };
    });
  }
};

// node_modules/vscode-languageserver-textdocument/lib/esm/main.js
var FullTextDocument2 = class _FullTextDocument {
  constructor(uri, languageId, version, content) {
    this._uri = uri;
    this._languageId = languageId;
    this._version = version;
    this._content = content;
    this._lineOffsets = void 0;
  }
  get uri() {
    return this._uri;
  }
  get languageId() {
    return this._languageId;
  }
  get version() {
    return this._version;
  }
  getText(range2) {
    if (range2) {
      const start = this.offsetAt(range2.start);
      const end = this.offsetAt(range2.end);
      return this._content.substring(start, end);
    }
    return this._content;
  }
  update(changes, version) {
    for (const change of changes) {
      if (_FullTextDocument.isIncremental(change)) {
        const range2 = getWellformedRange(change.range);
        const startOffset = this.offsetAt(range2.start);
        const endOffset = this.offsetAt(range2.end);
        this._content = this._content.substring(0, startOffset) + change.text + this._content.substring(endOffset, this._content.length);
        const startLine = Math.max(range2.start.line, 0);
        const endLine = Math.max(range2.end.line, 0);
        let lineOffsets = this._lineOffsets;
        const addedLineOffsets = computeLineOffsets(change.text, false, startOffset);
        if (endLine - startLine === addedLineOffsets.length) {
          for (let i = 0, len = addedLineOffsets.length; i < len; i++) {
            lineOffsets[i + startLine + 1] = addedLineOffsets[i];
          }
        } else {
          if (addedLineOffsets.length < 1e4) {
            lineOffsets.splice(startLine + 1, endLine - startLine, ...addedLineOffsets);
          } else {
            this._lineOffsets = lineOffsets = lineOffsets.slice(0, startLine + 1).concat(addedLineOffsets, lineOffsets.slice(endLine + 1));
          }
        }
        const diff = change.text.length - (endOffset - startOffset);
        if (diff !== 0) {
          for (let i = startLine + 1 + addedLineOffsets.length, len = lineOffsets.length; i < len; i++) {
            lineOffsets[i] = lineOffsets[i] + diff;
          }
        }
      } else if (_FullTextDocument.isFull(change)) {
        this._content = change.text;
        this._lineOffsets = void 0;
      } else {
        throw new Error("Unknown change event received");
      }
    }
    this._version = version;
  }
  getLineOffsets() {
    if (this._lineOffsets === void 0) {
      this._lineOffsets = computeLineOffsets(this._content, true);
    }
    return this._lineOffsets;
  }
  positionAt(offset) {
    offset = Math.max(Math.min(offset, this._content.length), 0);
    const lineOffsets = this.getLineOffsets();
    let low = 0, high = lineOffsets.length;
    if (high === 0) {
      return { line: 0, character: offset };
    }
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (lineOffsets[mid] > offset) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    const line = low - 1;
    offset = this.ensureBeforeEOL(offset, lineOffsets[line]);
    return { line, character: offset - lineOffsets[line] };
  }
  offsetAt(position) {
    const lineOffsets = this.getLineOffsets();
    if (position.line >= lineOffsets.length) {
      return this._content.length;
    } else if (position.line < 0) {
      return 0;
    }
    const lineOffset = lineOffsets[position.line];
    if (position.character <= 0) {
      return lineOffset;
    }
    const nextLineOffset = position.line + 1 < lineOffsets.length ? lineOffsets[position.line + 1] : this._content.length;
    const offset = Math.min(lineOffset + position.character, nextLineOffset);
    return this.ensureBeforeEOL(offset, lineOffset);
  }
  ensureBeforeEOL(offset, lineOffset) {
    while (offset > lineOffset && isEOL(this._content.charCodeAt(offset - 1))) {
      offset--;
    }
    return offset;
  }
  get lineCount() {
    return this.getLineOffsets().length;
  }
  static isIncremental(event) {
    const candidate = event;
    return candidate !== void 0 && candidate !== null && typeof candidate.text === "string" && candidate.range !== void 0 && (candidate.rangeLength === void 0 || typeof candidate.rangeLength === "number");
  }
  static isFull(event) {
    const candidate = event;
    return candidate !== void 0 && candidate !== null && typeof candidate.text === "string" && candidate.range === void 0 && candidate.rangeLength === void 0;
  }
};
var TextDocument2;
(function(TextDocument3) {
  function create2(uri, languageId, version, content) {
    return new FullTextDocument2(uri, languageId, version, content);
  }
  TextDocument3.create = create2;
  function update2(document, changes, version) {
    if (document instanceof FullTextDocument2) {
      document.update(changes, version);
      return document;
    } else {
      throw new Error("TextDocument.update: document must be created by TextDocument.create");
    }
  }
  TextDocument3.update = update2;
  function applyEdits(document, edits) {
    const text = document.getText();
    const sortedEdits = mergeSort(edits.map(getWellformedEdit), (a, b) => {
      const diff = a.range.start.line - b.range.start.line;
      if (diff === 0) {
        return a.range.start.character - b.range.start.character;
      }
      return diff;
    });
    let lastModifiedOffset = 0;
    const spans = [];
    for (const e of sortedEdits) {
      const startOffset = document.offsetAt(e.range.start);
      if (startOffset < lastModifiedOffset) {
        throw new Error("Overlapping edit");
      } else if (startOffset > lastModifiedOffset) {
        spans.push(text.substring(lastModifiedOffset, startOffset));
      }
      if (e.newText.length) {
        spans.push(e.newText);
      }
      lastModifiedOffset = document.offsetAt(e.range.end);
    }
    spans.push(text.substr(lastModifiedOffset));
    return spans.join("");
  }
  TextDocument3.applyEdits = applyEdits;
})(TextDocument2 || (TextDocument2 = {}));
function mergeSort(data, compare) {
  if (data.length <= 1) {
    return data;
  }
  const p = data.length / 2 | 0;
  const left = data.slice(0, p);
  const right = data.slice(p);
  mergeSort(left, compare);
  mergeSort(right, compare);
  let leftIdx = 0;
  let rightIdx = 0;
  let i = 0;
  while (leftIdx < left.length && rightIdx < right.length) {
    const ret = compare(left[leftIdx], right[rightIdx]);
    if (ret <= 0) {
      data[i++] = left[leftIdx++];
    } else {
      data[i++] = right[rightIdx++];
    }
  }
  while (leftIdx < left.length) {
    data[i++] = left[leftIdx++];
  }
  while (rightIdx < right.length) {
    data[i++] = right[rightIdx++];
  }
  return data;
}
function computeLineOffsets(text, isAtLineStart, textOffset = 0) {
  const result2 = isAtLineStart ? [textOffset] : [];
  for (let i = 0; i < text.length; i++) {
    const ch = text.charCodeAt(i);
    if (isEOL(ch)) {
      if (ch === 13 && i + 1 < text.length && text.charCodeAt(i + 1) === 10) {
        i++;
      }
      result2.push(textOffset + i + 1);
    }
  }
  return result2;
}
function isEOL(char) {
  return char === 13 || char === 10;
}
function getWellformedRange(range2) {
  const start = range2.start;
  const end = range2.end;
  if (start.line > end.line || start.line === end.line && start.character > end.character) {
    return { start: end, end: start };
  }
  return range2;
}
function getWellformedEdit(textEdit) {
  const range2 = getWellformedRange(textEdit.range);
  if (range2 !== textEdit.range) {
    return { newText: textEdit.newText, range: range2 };
  }
  return textEdit;
}

// node_modules/vscode-uri/lib/esm/index.mjs
var LIB;
(() => {
  "use strict";
  var t = { 470: (t2) => {
    function e2(t3) {
      if ("string" != typeof t3) throw new TypeError("Path must be a string. Received " + JSON.stringify(t3));
    }
    function r2(t3, e3) {
      for (var r3, n3 = "", i = 0, o = -1, s = 0, h = 0; h <= t3.length; ++h) {
        if (h < t3.length) r3 = t3.charCodeAt(h);
        else {
          if (47 === r3) break;
          r3 = 47;
        }
        if (47 === r3) {
          if (o === h - 1 || 1 === s) ;
          else if (o !== h - 1 && 2 === s) {
            if (n3.length < 2 || 2 !== i || 46 !== n3.charCodeAt(n3.length - 1) || 46 !== n3.charCodeAt(n3.length - 2)) {
              if (n3.length > 2) {
                var a = n3.lastIndexOf("/");
                if (a !== n3.length - 1) {
                  -1 === a ? (n3 = "", i = 0) : i = (n3 = n3.slice(0, a)).length - 1 - n3.lastIndexOf("/"), o = h, s = 0;
                  continue;
                }
              } else if (2 === n3.length || 1 === n3.length) {
                n3 = "", i = 0, o = h, s = 0;
                continue;
              }
            }
            e3 && (n3.length > 0 ? n3 += "/.." : n3 = "..", i = 2);
          } else n3.length > 0 ? n3 += "/" + t3.slice(o + 1, h) : n3 = t3.slice(o + 1, h), i = h - o - 1;
          o = h, s = 0;
        } else 46 === r3 && -1 !== s ? ++s : s = -1;
      }
      return n3;
    }
    var n2 = { resolve: function() {
      for (var t3, n3 = "", i = false, o = arguments.length - 1; o >= -1 && !i; o--) {
        var s;
        o >= 0 ? s = arguments[o] : (void 0 === t3 && (t3 = process.cwd()), s = t3), e2(s), 0 !== s.length && (n3 = s + "/" + n3, i = 47 === s.charCodeAt(0));
      }
      return n3 = r2(n3, !i), i ? n3.length > 0 ? "/" + n3 : "/" : n3.length > 0 ? n3 : ".";
    }, normalize: function(t3) {
      if (e2(t3), 0 === t3.length) return ".";
      var n3 = 47 === t3.charCodeAt(0), i = 47 === t3.charCodeAt(t3.length - 1);
      return 0 !== (t3 = r2(t3, !n3)).length || n3 || (t3 = "."), t3.length > 0 && i && (t3 += "/"), n3 ? "/" + t3 : t3;
    }, isAbsolute: function(t3) {
      return e2(t3), t3.length > 0 && 47 === t3.charCodeAt(0);
    }, join: function() {
      if (0 === arguments.length) return ".";
      for (var t3, r3 = 0; r3 < arguments.length; ++r3) {
        var i = arguments[r3];
        e2(i), i.length > 0 && (void 0 === t3 ? t3 = i : t3 += "/" + i);
      }
      return void 0 === t3 ? "." : n2.normalize(t3);
    }, relative: function(t3, r3) {
      if (e2(t3), e2(r3), t3 === r3) return "";
      if ((t3 = n2.resolve(t3)) === (r3 = n2.resolve(r3))) return "";
      for (var i = 1; i < t3.length && 47 === t3.charCodeAt(i); ++i) ;
      for (var o = t3.length, s = o - i, h = 1; h < r3.length && 47 === r3.charCodeAt(h); ++h) ;
      for (var a = r3.length - h, c = s < a ? s : a, f = -1, u = 0; u <= c; ++u) {
        if (u === c) {
          if (a > c) {
            if (47 === r3.charCodeAt(h + u)) return r3.slice(h + u + 1);
            if (0 === u) return r3.slice(h + u);
          } else s > c && (47 === t3.charCodeAt(i + u) ? f = u : 0 === u && (f = 0));
          break;
        }
        var l = t3.charCodeAt(i + u);
        if (l !== r3.charCodeAt(h + u)) break;
        47 === l && (f = u);
      }
      var g = "";
      for (u = i + f + 1; u <= o; ++u) u !== o && 47 !== t3.charCodeAt(u) || (0 === g.length ? g += ".." : g += "/..");
      return g.length > 0 ? g + r3.slice(h + f) : (h += f, 47 === r3.charCodeAt(h) && ++h, r3.slice(h));
    }, _makeLong: function(t3) {
      return t3;
    }, dirname: function(t3) {
      if (e2(t3), 0 === t3.length) return ".";
      for (var r3 = t3.charCodeAt(0), n3 = 47 === r3, i = -1, o = true, s = t3.length - 1; s >= 1; --s) if (47 === (r3 = t3.charCodeAt(s))) {
        if (!o) {
          i = s;
          break;
        }
      } else o = false;
      return -1 === i ? n3 ? "/" : "." : n3 && 1 === i ? "//" : t3.slice(0, i);
    }, basename: function(t3, r3) {
      if (void 0 !== r3 && "string" != typeof r3) throw new TypeError('"ext" argument must be a string');
      e2(t3);
      var n3, i = 0, o = -1, s = true;
      if (void 0 !== r3 && r3.length > 0 && r3.length <= t3.length) {
        if (r3.length === t3.length && r3 === t3) return "";
        var h = r3.length - 1, a = -1;
        for (n3 = t3.length - 1; n3 >= 0; --n3) {
          var c = t3.charCodeAt(n3);
          if (47 === c) {
            if (!s) {
              i = n3 + 1;
              break;
            }
          } else -1 === a && (s = false, a = n3 + 1), h >= 0 && (c === r3.charCodeAt(h) ? -1 == --h && (o = n3) : (h = -1, o = a));
        }
        return i === o ? o = a : -1 === o && (o = t3.length), t3.slice(i, o);
      }
      for (n3 = t3.length - 1; n3 >= 0; --n3) if (47 === t3.charCodeAt(n3)) {
        if (!s) {
          i = n3 + 1;
          break;
        }
      } else -1 === o && (s = false, o = n3 + 1);
      return -1 === o ? "" : t3.slice(i, o);
    }, extname: function(t3) {
      e2(t3);
      for (var r3 = -1, n3 = 0, i = -1, o = true, s = 0, h = t3.length - 1; h >= 0; --h) {
        var a = t3.charCodeAt(h);
        if (47 !== a) -1 === i && (o = false, i = h + 1), 46 === a ? -1 === r3 ? r3 = h : 1 !== s && (s = 1) : -1 !== r3 && (s = -1);
        else if (!o) {
          n3 = h + 1;
          break;
        }
      }
      return -1 === r3 || -1 === i || 0 === s || 1 === s && r3 === i - 1 && r3 === n3 + 1 ? "" : t3.slice(r3, i);
    }, format: function(t3) {
      if (null === t3 || "object" != typeof t3) throw new TypeError('The "pathObject" argument must be of type Object. Received type ' + typeof t3);
      return function(t4, e3) {
        var r3 = e3.dir || e3.root, n3 = e3.base || (e3.name || "") + (e3.ext || "");
        return r3 ? r3 === e3.root ? r3 + n3 : r3 + "/" + n3 : n3;
      }(0, t3);
    }, parse: function(t3) {
      e2(t3);
      var r3 = { root: "", dir: "", base: "", ext: "", name: "" };
      if (0 === t3.length) return r3;
      var n3, i = t3.charCodeAt(0), o = 47 === i;
      o ? (r3.root = "/", n3 = 1) : n3 = 0;
      for (var s = -1, h = 0, a = -1, c = true, f = t3.length - 1, u = 0; f >= n3; --f) if (47 !== (i = t3.charCodeAt(f))) -1 === a && (c = false, a = f + 1), 46 === i ? -1 === s ? s = f : 1 !== u && (u = 1) : -1 !== s && (u = -1);
      else if (!c) {
        h = f + 1;
        break;
      }
      return -1 === s || -1 === a || 0 === u || 1 === u && s === a - 1 && s === h + 1 ? -1 !== a && (r3.base = r3.name = 0 === h && o ? t3.slice(1, a) : t3.slice(h, a)) : (0 === h && o ? (r3.name = t3.slice(1, s), r3.base = t3.slice(1, a)) : (r3.name = t3.slice(h, s), r3.base = t3.slice(h, a)), r3.ext = t3.slice(s, a)), h > 0 ? r3.dir = t3.slice(0, h - 1) : o && (r3.dir = "/"), r3;
    }, sep: "/", delimiter: ":", win32: null, posix: null };
    n2.posix = n2, t2.exports = n2;
  } }, e = {};
  function r(n2) {
    var i = e[n2];
    if (void 0 !== i) return i.exports;
    var o = e[n2] = { exports: {} };
    return t[n2](o, o.exports, r), o.exports;
  }
  r.d = (t2, e2) => {
    for (var n2 in e2) r.o(e2, n2) && !r.o(t2, n2) && Object.defineProperty(t2, n2, { enumerable: true, get: e2[n2] });
  }, r.o = (t2, e2) => Object.prototype.hasOwnProperty.call(t2, e2), r.r = (t2) => {
    "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(t2, Symbol.toStringTag, { value: "Module" }), Object.defineProperty(t2, "__esModule", { value: true });
  };
  var n = {};
  (() => {
    let t2;
    if (r.r(n), r.d(n, { URI: () => f, Utils: () => P }), "object" == typeof process) t2 = "win32" === process.platform;
    else if ("object" == typeof navigator) {
      let e3 = navigator.userAgent;
      t2 = e3.indexOf("Windows") >= 0;
    }
    const e2 = /^\w[\w\d+.-]*$/, i = /^\//, o = /^\/\//;
    function s(t3, r2) {
      if (!t3.scheme && r2) throw new Error(`[UriError]: Scheme is missing: {scheme: "", authority: "${t3.authority}", path: "${t3.path}", query: "${t3.query}", fragment: "${t3.fragment}"}`);
      if (t3.scheme && !e2.test(t3.scheme)) throw new Error("[UriError]: Scheme contains illegal characters.");
      if (t3.path) {
        if (t3.authority) {
          if (!i.test(t3.path)) throw new Error('[UriError]: If a URI contains an authority component, then the path component must either be empty or begin with a slash ("/") character');
        } else if (o.test(t3.path)) throw new Error('[UriError]: If a URI does not contain an authority component, then the path cannot begin with two slash characters ("//")');
      }
    }
    const h = "", a = "/", c = /^(([^:/?#]+?):)?(\/\/([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?/;
    class f {
      static isUri(t3) {
        return t3 instanceof f || !!t3 && "string" == typeof t3.authority && "string" == typeof t3.fragment && "string" == typeof t3.path && "string" == typeof t3.query && "string" == typeof t3.scheme && "string" == typeof t3.fsPath && "function" == typeof t3.with && "function" == typeof t3.toString;
      }
      scheme;
      authority;
      path;
      query;
      fragment;
      constructor(t3, e3, r2, n2, i2, o2 = false) {
        "object" == typeof t3 ? (this.scheme = t3.scheme || h, this.authority = t3.authority || h, this.path = t3.path || h, this.query = t3.query || h, this.fragment = t3.fragment || h) : (this.scheme = /* @__PURE__ */ function(t4, e4) {
          return t4 || e4 ? t4 : "file";
        }(t3, o2), this.authority = e3 || h, this.path = function(t4, e4) {
          switch (t4) {
            case "https":
            case "http":
            case "file":
              e4 ? e4[0] !== a && (e4 = a + e4) : e4 = a;
          }
          return e4;
        }(this.scheme, r2 || h), this.query = n2 || h, this.fragment = i2 || h, s(this, o2));
      }
      get fsPath() {
        return m(this, false);
      }
      with(t3) {
        if (!t3) return this;
        let { scheme: e3, authority: r2, path: n2, query: i2, fragment: o2 } = t3;
        return void 0 === e3 ? e3 = this.scheme : null === e3 && (e3 = h), void 0 === r2 ? r2 = this.authority : null === r2 && (r2 = h), void 0 === n2 ? n2 = this.path : null === n2 && (n2 = h), void 0 === i2 ? i2 = this.query : null === i2 && (i2 = h), void 0 === o2 ? o2 = this.fragment : null === o2 && (o2 = h), e3 === this.scheme && r2 === this.authority && n2 === this.path && i2 === this.query && o2 === this.fragment ? this : new l(e3, r2, n2, i2, o2);
      }
      static parse(t3, e3 = false) {
        const r2 = c.exec(t3);
        return r2 ? new l(r2[2] || h, C(r2[4] || h), C(r2[5] || h), C(r2[7] || h), C(r2[9] || h), e3) : new l(h, h, h, h, h);
      }
      static file(e3) {
        let r2 = h;
        if (t2 && (e3 = e3.replace(/\\/g, a)), e3[0] === a && e3[1] === a) {
          const t3 = e3.indexOf(a, 2);
          -1 === t3 ? (r2 = e3.substring(2), e3 = a) : (r2 = e3.substring(2, t3), e3 = e3.substring(t3) || a);
        }
        return new l("file", r2, e3, h, h);
      }
      static from(t3) {
        const e3 = new l(t3.scheme, t3.authority, t3.path, t3.query, t3.fragment);
        return s(e3, true), e3;
      }
      toString(t3 = false) {
        return y(this, t3);
      }
      toJSON() {
        return this;
      }
      static revive(t3) {
        if (t3) {
          if (t3 instanceof f) return t3;
          {
            const e3 = new l(t3);
            return e3._formatted = t3.external, e3._fsPath = t3._sep === u ? t3.fsPath : null, e3;
          }
        }
        return t3;
      }
    }
    const u = t2 ? 1 : void 0;
    class l extends f {
      _formatted = null;
      _fsPath = null;
      get fsPath() {
        return this._fsPath || (this._fsPath = m(this, false)), this._fsPath;
      }
      toString(t3 = false) {
        return t3 ? y(this, true) : (this._formatted || (this._formatted = y(this, false)), this._formatted);
      }
      toJSON() {
        const t3 = { $mid: 1 };
        return this._fsPath && (t3.fsPath = this._fsPath, t3._sep = u), this._formatted && (t3.external = this._formatted), this.path && (t3.path = this.path), this.scheme && (t3.scheme = this.scheme), this.authority && (t3.authority = this.authority), this.query && (t3.query = this.query), this.fragment && (t3.fragment = this.fragment), t3;
      }
    }
    const g = { 58: "%3A", 47: "%2F", 63: "%3F", 35: "%23", 91: "%5B", 93: "%5D", 64: "%40", 33: "%21", 36: "%24", 38: "%26", 39: "%27", 40: "%28", 41: "%29", 42: "%2A", 43: "%2B", 44: "%2C", 59: "%3B", 61: "%3D", 32: "%20" };
    function d(t3, e3, r2) {
      let n2, i2 = -1;
      for (let o2 = 0; o2 < t3.length; o2++) {
        const s2 = t3.charCodeAt(o2);
        if (s2 >= 97 && s2 <= 122 || s2 >= 65 && s2 <= 90 || s2 >= 48 && s2 <= 57 || 45 === s2 || 46 === s2 || 95 === s2 || 126 === s2 || e3 && 47 === s2 || r2 && 91 === s2 || r2 && 93 === s2 || r2 && 58 === s2) -1 !== i2 && (n2 += encodeURIComponent(t3.substring(i2, o2)), i2 = -1), void 0 !== n2 && (n2 += t3.charAt(o2));
        else {
          void 0 === n2 && (n2 = t3.substr(0, o2));
          const e4 = g[s2];
          void 0 !== e4 ? (-1 !== i2 && (n2 += encodeURIComponent(t3.substring(i2, o2)), i2 = -1), n2 += e4) : -1 === i2 && (i2 = o2);
        }
      }
      return -1 !== i2 && (n2 += encodeURIComponent(t3.substring(i2))), void 0 !== n2 ? n2 : t3;
    }
    function p(t3) {
      let e3;
      for (let r2 = 0; r2 < t3.length; r2++) {
        const n2 = t3.charCodeAt(r2);
        35 === n2 || 63 === n2 ? (void 0 === e3 && (e3 = t3.substr(0, r2)), e3 += g[n2]) : void 0 !== e3 && (e3 += t3[r2]);
      }
      return void 0 !== e3 ? e3 : t3;
    }
    function m(e3, r2) {
      let n2;
      return n2 = e3.authority && e3.path.length > 1 && "file" === e3.scheme ? `//${e3.authority}${e3.path}` : 47 === e3.path.charCodeAt(0) && (e3.path.charCodeAt(1) >= 65 && e3.path.charCodeAt(1) <= 90 || e3.path.charCodeAt(1) >= 97 && e3.path.charCodeAt(1) <= 122) && 58 === e3.path.charCodeAt(2) ? r2 ? e3.path.substr(1) : e3.path[1].toLowerCase() + e3.path.substr(2) : e3.path, t2 && (n2 = n2.replace(/\//g, "\\")), n2;
    }
    function y(t3, e3) {
      const r2 = e3 ? p : d;
      let n2 = "", { scheme: i2, authority: o2, path: s2, query: h2, fragment: c2 } = t3;
      if (i2 && (n2 += i2, n2 += ":"), (o2 || "file" === i2) && (n2 += a, n2 += a), o2) {
        let t4 = o2.indexOf("@");
        if (-1 !== t4) {
          const e4 = o2.substr(0, t4);
          o2 = o2.substr(t4 + 1), t4 = e4.lastIndexOf(":"), -1 === t4 ? n2 += r2(e4, false, false) : (n2 += r2(e4.substr(0, t4), false, false), n2 += ":", n2 += r2(e4.substr(t4 + 1), false, true)), n2 += "@";
        }
        o2 = o2.toLowerCase(), t4 = o2.lastIndexOf(":"), -1 === t4 ? n2 += r2(o2, false, true) : (n2 += r2(o2.substr(0, t4), false, true), n2 += o2.substr(t4));
      }
      if (s2) {
        if (s2.length >= 3 && 47 === s2.charCodeAt(0) && 58 === s2.charCodeAt(2)) {
          const t4 = s2.charCodeAt(1);
          t4 >= 65 && t4 <= 90 && (s2 = `/${String.fromCharCode(t4 + 32)}:${s2.substr(3)}`);
        } else if (s2.length >= 2 && 58 === s2.charCodeAt(1)) {
          const t4 = s2.charCodeAt(0);
          t4 >= 65 && t4 <= 90 && (s2 = `${String.fromCharCode(t4 + 32)}:${s2.substr(2)}`);
        }
        n2 += r2(s2, true, false);
      }
      return h2 && (n2 += "?", n2 += r2(h2, false, false)), c2 && (n2 += "#", n2 += e3 ? c2 : d(c2, false, false)), n2;
    }
    function v(t3) {
      try {
        return decodeURIComponent(t3);
      } catch {
        return t3.length > 3 ? t3.substr(0, 3) + v(t3.substr(3)) : t3;
      }
    }
    const b = /(%[0-9A-Za-z][0-9A-Za-z])+/g;
    function C(t3) {
      return t3.match(b) ? t3.replace(b, (t4) => v(t4)) : t3;
    }
    var A = r(470);
    const w = A.posix || A, x = "/";
    var P;
    !function(t3) {
      t3.joinPath = function(t4, ...e3) {
        return t4.with({ path: w.join(t4.path, ...e3) });
      }, t3.resolvePath = function(t4, ...e3) {
        let r2 = t4.path, n2 = false;
        r2[0] !== x && (r2 = x + r2, n2 = true);
        let i2 = w.resolve(r2, ...e3);
        return n2 && i2[0] === x && !t4.authority && (i2 = i2.substring(1)), t4.with({ path: i2 });
      }, t3.dirname = function(t4) {
        if (0 === t4.path.length || t4.path === x) return t4;
        let e3 = w.dirname(t4.path);
        return 1 === e3.length && 46 === e3.charCodeAt(0) && (e3 = ""), t4.with({ path: e3 });
      }, t3.basename = function(t4) {
        return w.basename(t4.path);
      }, t3.extname = function(t4) {
        return w.extname(t4.path);
      };
    }(P || (P = {}));
  })(), LIB = n;
})();
var { URI: URI2, Utils } = LIB;

// node_modules/langium/lib/utils/uri-utils.js
var UriUtils;
(function(UriUtils2) {
  UriUtils2.basename = Utils.basename;
  UriUtils2.dirname = Utils.dirname;
  UriUtils2.extname = Utils.extname;
  UriUtils2.joinPath = Utils.joinPath;
  UriUtils2.resolvePath = Utils.resolvePath;
  function equals(a, b) {
    return (a === null || a === void 0 ? void 0 : a.toString()) === (b === null || b === void 0 ? void 0 : b.toString());
  }
  UriUtils2.equals = equals;
  function relative(from, to) {
    const fromPath = typeof from === "string" ? from : from.path;
    const toPath2 = typeof to === "string" ? to : to.path;
    const fromParts = fromPath.split("/").filter((e) => e.length > 0);
    const toParts = toPath2.split("/").filter((e) => e.length > 0);
    let i = 0;
    for (; i < fromParts.length; i++) {
      if (fromParts[i] !== toParts[i]) {
        break;
      }
    }
    const backPart = "../".repeat(fromParts.length - i);
    const toPart = toParts.slice(i).join("/");
    return backPart + toPart;
  }
  UriUtils2.relative = relative;
  function normalize(uri) {
    return URI2.parse(uri.toString()).toString();
  }
  UriUtils2.normalize = normalize;
})(UriUtils || (UriUtils = {}));

// node_modules/langium/lib/workspace/documents.js
var DocumentState;
(function(DocumentState2) {
  DocumentState2[DocumentState2["Changed"] = 0] = "Changed";
  DocumentState2[DocumentState2["Parsed"] = 1] = "Parsed";
  DocumentState2[DocumentState2["IndexedContent"] = 2] = "IndexedContent";
  DocumentState2[DocumentState2["ComputedScopes"] = 3] = "ComputedScopes";
  DocumentState2[DocumentState2["Linked"] = 4] = "Linked";
  DocumentState2[DocumentState2["IndexedReferences"] = 5] = "IndexedReferences";
  DocumentState2[DocumentState2["Validated"] = 6] = "Validated";
})(DocumentState || (DocumentState = {}));
var DefaultLangiumDocumentFactory = class {
  constructor(services) {
    this.serviceRegistry = services.ServiceRegistry;
    this.textDocuments = services.workspace.TextDocuments;
    this.fileSystemProvider = services.workspace.FileSystemProvider;
  }
  async fromUri(uri, cancellationToken = cancellation_exports.CancellationToken.None) {
    const content = await this.fileSystemProvider.readFile(uri);
    return this.createAsync(uri, content, cancellationToken);
  }
  fromTextDocument(textDocument, uri, token) {
    uri = uri !== null && uri !== void 0 ? uri : URI2.parse(textDocument.uri);
    if (cancellation_exports.CancellationToken.is(token)) {
      return this.createAsync(uri, textDocument, token);
    } else {
      return this.create(uri, textDocument, token);
    }
  }
  fromString(text, uri, token) {
    if (cancellation_exports.CancellationToken.is(token)) {
      return this.createAsync(uri, text, token);
    } else {
      return this.create(uri, text, token);
    }
  }
  fromModel(model, uri) {
    return this.create(uri, { $model: model });
  }
  create(uri, content, options) {
    if (typeof content === "string") {
      const parseResult = this.parse(uri, content, options);
      return this.createLangiumDocument(parseResult, uri, void 0, content);
    } else if ("$model" in content) {
      const parseResult = { value: content.$model, parserErrors: [], lexerErrors: [] };
      return this.createLangiumDocument(parseResult, uri);
    } else {
      const parseResult = this.parse(uri, content.getText(), options);
      return this.createLangiumDocument(parseResult, uri, content);
    }
  }
  async createAsync(uri, content, cancelToken) {
    if (typeof content === "string") {
      const parseResult = await this.parseAsync(uri, content, cancelToken);
      return this.createLangiumDocument(parseResult, uri, void 0, content);
    } else {
      const parseResult = await this.parseAsync(uri, content.getText(), cancelToken);
      return this.createLangiumDocument(parseResult, uri, content);
    }
  }
  /**
   * Create a LangiumDocument from a given parse result.
   *
   * A TextDocument is created on demand if it is not provided as argument here. Usually this
   * should not be necessary because the main purpose of the TextDocument is to convert between
   * text ranges and offsets, which is done solely in LSP request handling.
   *
   * With the introduction of {@link update} below this method is supposed to be mainly called
   * during workspace initialization and on addition/recognition of new files, while changes in
   * existing documents are processed via {@link update}.
   */
  createLangiumDocument(parseResult, uri, textDocument, text) {
    let document;
    if (textDocument) {
      document = {
        parseResult,
        uri,
        state: DocumentState.Parsed,
        references: [],
        textDocument
      };
    } else {
      const textDocumentGetter = this.createTextDocumentGetter(uri, text);
      document = {
        parseResult,
        uri,
        state: DocumentState.Parsed,
        references: [],
        get textDocument() {
          return textDocumentGetter();
        }
      };
    }
    parseResult.value.$document = document;
    return document;
  }
  async update(document, cancellationToken) {
    var _a, _b;
    const oldText = (_a = document.parseResult.value.$cstNode) === null || _a === void 0 ? void 0 : _a.root.fullText;
    const textDocument = (_b = this.textDocuments) === null || _b === void 0 ? void 0 : _b.get(document.uri.toString());
    const text = textDocument ? textDocument.getText() : await this.fileSystemProvider.readFile(document.uri);
    if (textDocument) {
      Object.defineProperty(document, "textDocument", {
        value: textDocument
      });
    } else {
      const textDocumentGetter = this.createTextDocumentGetter(document.uri, text);
      Object.defineProperty(document, "textDocument", {
        get: textDocumentGetter
      });
    }
    if (oldText !== text) {
      document.parseResult = await this.parseAsync(document.uri, text, cancellationToken);
      document.parseResult.value.$document = document;
    }
    document.state = DocumentState.Parsed;
    return document;
  }
  parse(uri, text, options) {
    const services = this.serviceRegistry.getServices(uri);
    return services.parser.LangiumParser.parse(text, options);
  }
  parseAsync(uri, text, cancellationToken) {
    const services = this.serviceRegistry.getServices(uri);
    return services.parser.AsyncParser.parse(text, cancellationToken);
  }
  createTextDocumentGetter(uri, text) {
    const serviceRegistry = this.serviceRegistry;
    let textDoc = void 0;
    return () => {
      return textDoc !== null && textDoc !== void 0 ? textDoc : textDoc = TextDocument2.create(uri.toString(), serviceRegistry.getServices(uri).LanguageMetaData.languageId, 0, text !== null && text !== void 0 ? text : "");
    };
  }
};
var DefaultLangiumDocuments = class {
  constructor(services) {
    this.documentMap = /* @__PURE__ */ new Map();
    this.langiumDocumentFactory = services.workspace.LangiumDocumentFactory;
    this.serviceRegistry = services.ServiceRegistry;
  }
  get all() {
    return stream(this.documentMap.values());
  }
  addDocument(document) {
    const uriString = document.uri.toString();
    if (this.documentMap.has(uriString)) {
      throw new Error(`A document with the URI '${uriString}' is already present.`);
    }
    this.documentMap.set(uriString, document);
  }
  getDocument(uri) {
    const uriString = uri.toString();
    return this.documentMap.get(uriString);
  }
  async getOrCreateDocument(uri, cancellationToken) {
    let document = this.getDocument(uri);
    if (document) {
      return document;
    }
    document = await this.langiumDocumentFactory.fromUri(uri, cancellationToken);
    this.addDocument(document);
    return document;
  }
  createDocument(uri, text, cancellationToken) {
    if (cancellationToken) {
      return this.langiumDocumentFactory.fromString(text, uri, cancellationToken).then((document) => {
        this.addDocument(document);
        return document;
      });
    } else {
      const document = this.langiumDocumentFactory.fromString(text, uri);
      this.addDocument(document);
      return document;
    }
  }
  hasDocument(uri) {
    return this.documentMap.has(uri.toString());
  }
  invalidateDocument(uri) {
    const uriString = uri.toString();
    const langiumDoc = this.documentMap.get(uriString);
    if (langiumDoc) {
      const linker = this.serviceRegistry.getServices(uri).references.Linker;
      linker.unlink(langiumDoc);
      langiumDoc.state = DocumentState.Changed;
      langiumDoc.precomputedScopes = void 0;
      langiumDoc.diagnostics = void 0;
    }
    return langiumDoc;
  }
  deleteDocument(uri) {
    const uriString = uri.toString();
    const langiumDoc = this.documentMap.get(uriString);
    if (langiumDoc) {
      langiumDoc.state = DocumentState.Changed;
      this.documentMap.delete(uriString);
    }
    return langiumDoc;
  }
};

// node_modules/langium/lib/references/linker.js
var ref_resolving = Symbol("ref_resolving");
var DefaultLinker = class {
  constructor(services) {
    this.reflection = services.shared.AstReflection;
    this.langiumDocuments = () => services.shared.workspace.LangiumDocuments;
    this.scopeProvider = services.references.ScopeProvider;
    this.astNodeLocator = services.workspace.AstNodeLocator;
  }
  async link(document, cancelToken = cancellation_exports.CancellationToken.None) {
    for (const node of streamAst(document.parseResult.value)) {
      await interruptAndCheck(cancelToken);
      streamReferences(node).forEach((ref) => this.doLink(ref, document));
    }
  }
  doLink(refInfo, document) {
    var _a;
    const ref = refInfo.reference;
    if (ref._ref === void 0) {
      ref._ref = ref_resolving;
      try {
        const description = this.getCandidate(refInfo);
        if (isLinkingError(description)) {
          ref._ref = description;
        } else {
          ref._nodeDescription = description;
          if (this.langiumDocuments().hasDocument(description.documentUri)) {
            const linkedNode = this.loadAstNode(description);
            ref._ref = linkedNode !== null && linkedNode !== void 0 ? linkedNode : this.createLinkingError(refInfo, description);
          } else {
            ref._ref = void 0;
          }
        }
      } catch (err) {
        console.error(`An error occurred while resolving reference to '${ref.$refText}':`, err);
        const errorMessage = (_a = err.message) !== null && _a !== void 0 ? _a : String(err);
        ref._ref = Object.assign(Object.assign({}, refInfo), { message: `An error occurred while resolving reference to '${ref.$refText}': ${errorMessage}` });
      }
      document.references.push(ref);
    }
  }
  unlink(document) {
    for (const ref of document.references) {
      delete ref._ref;
      delete ref._nodeDescription;
    }
    document.references = [];
  }
  getCandidate(refInfo) {
    const scope = this.scopeProvider.getScope(refInfo);
    const description = scope.getElement(refInfo.reference.$refText);
    return description !== null && description !== void 0 ? description : this.createLinkingError(refInfo);
  }
  buildReference(node, property2, refNode, refText) {
    const linker = this;
    const reference = {
      $refNode: refNode,
      $refText: refText,
      get ref() {
        var _a;
        if (isAstNode(this._ref)) {
          return this._ref;
        } else if (isAstNodeDescription(this._nodeDescription)) {
          const linkedNode = linker.loadAstNode(this._nodeDescription);
          this._ref = linkedNode !== null && linkedNode !== void 0 ? linkedNode : linker.createLinkingError({ reference, container: node, property: property2 }, this._nodeDescription);
        } else if (this._ref === void 0) {
          this._ref = ref_resolving;
          const document = findRootNode(node).$document;
          const refData = linker.getLinkedNode({ reference, container: node, property: property2 });
          if (refData.error && document && document.state < DocumentState.ComputedScopes) {
            return this._ref = void 0;
          }
          this._ref = (_a = refData.node) !== null && _a !== void 0 ? _a : refData.error;
          this._nodeDescription = refData.descr;
          document === null || document === void 0 ? void 0 : document.references.push(this);
        } else if (this._ref === ref_resolving) {
          throw new Error(`Cyclic reference resolution detected: ${linker.astNodeLocator.getAstNodePath(node)}/${property2} (symbol '${refText}')`);
        }
        return isAstNode(this._ref) ? this._ref : void 0;
      },
      get $nodeDescription() {
        return this._nodeDescription;
      },
      get error() {
        return isLinkingError(this._ref) ? this._ref : void 0;
      }
    };
    return reference;
  }
  getLinkedNode(refInfo) {
    var _a;
    try {
      const description = this.getCandidate(refInfo);
      if (isLinkingError(description)) {
        return { error: description };
      }
      const linkedNode = this.loadAstNode(description);
      if (linkedNode) {
        return { node: linkedNode, descr: description };
      } else {
        return {
          descr: description,
          error: this.createLinkingError(refInfo, description)
        };
      }
    } catch (err) {
      console.error(`An error occurred while resolving reference to '${refInfo.reference.$refText}':`, err);
      const errorMessage = (_a = err.message) !== null && _a !== void 0 ? _a : String(err);
      return {
        error: Object.assign(Object.assign({}, refInfo), { message: `An error occurred while resolving reference to '${refInfo.reference.$refText}': ${errorMessage}` })
      };
    }
  }
  loadAstNode(nodeDescription) {
    if (nodeDescription.node) {
      return nodeDescription.node;
    }
    const doc = this.langiumDocuments().getDocument(nodeDescription.documentUri);
    if (!doc) {
      return void 0;
    }
    return this.astNodeLocator.getAstNode(doc.parseResult.value, nodeDescription.path);
  }
  createLinkingError(refInfo, targetDescription) {
    const document = findRootNode(refInfo.container).$document;
    if (document && document.state < DocumentState.ComputedScopes) {
      console.warn(`Attempted reference resolution before document reached ComputedScopes state (${document.uri}).`);
    }
    const referenceType = this.reflection.getReferenceType(refInfo);
    return Object.assign(Object.assign({}, refInfo), { message: `Could not resolve reference to ${referenceType} named '${refInfo.reference.$refText}'.`, targetDescription });
  }
};

// node_modules/langium/lib/references/name-provider.js
function isNamed(node) {
  return typeof node.name === "string";
}
var DefaultNameProvider = class {
  getName(node) {
    if (isNamed(node)) {
      return node.name;
    }
    return void 0;
  }
  getNameNode(node) {
    return findNodeForProperty(node.$cstNode, "name");
  }
};

// node_modules/langium/lib/references/references.js
var DefaultReferences = class {
  constructor(services) {
    this.nameProvider = services.references.NameProvider;
    this.index = services.shared.workspace.IndexManager;
    this.nodeLocator = services.workspace.AstNodeLocator;
  }
  findDeclaration(sourceCstNode) {
    if (sourceCstNode) {
      const assignment = findAssignment(sourceCstNode);
      const nodeElem = sourceCstNode.astNode;
      if (assignment && nodeElem) {
        const reference = nodeElem[assignment.feature];
        if (isReference(reference)) {
          return reference.ref;
        } else if (Array.isArray(reference)) {
          for (const ref of reference) {
            if (isReference(ref) && ref.$refNode && ref.$refNode.offset <= sourceCstNode.offset && ref.$refNode.end >= sourceCstNode.end) {
              return ref.ref;
            }
          }
        }
      }
      if (nodeElem) {
        const nameNode = this.nameProvider.getNameNode(nodeElem);
        if (nameNode && (nameNode === sourceCstNode || isChildNode(sourceCstNode, nameNode))) {
          return nodeElem;
        }
      }
    }
    return void 0;
  }
  findDeclarationNode(sourceCstNode) {
    const astNode = this.findDeclaration(sourceCstNode);
    if (astNode === null || astNode === void 0 ? void 0 : astNode.$cstNode) {
      const targetNode = this.nameProvider.getNameNode(astNode);
      return targetNode !== null && targetNode !== void 0 ? targetNode : astNode.$cstNode;
    }
    return void 0;
  }
  findReferences(targetNode, options) {
    const refs = [];
    if (options.includeDeclaration) {
      const ref = this.getReferenceToSelf(targetNode);
      if (ref) {
        refs.push(ref);
      }
    }
    let indexReferences = this.index.findAllReferences(targetNode, this.nodeLocator.getAstNodePath(targetNode));
    if (options.documentUri) {
      indexReferences = indexReferences.filter((ref) => UriUtils.equals(ref.sourceUri, options.documentUri));
    }
    refs.push(...indexReferences);
    return stream(refs);
  }
  getReferenceToSelf(targetNode) {
    const nameNode = this.nameProvider.getNameNode(targetNode);
    if (nameNode) {
      const doc = getDocument(targetNode);
      const path = this.nodeLocator.getAstNodePath(targetNode);
      return {
        sourceUri: doc.uri,
        sourcePath: path,
        targetUri: doc.uri,
        targetPath: path,
        segment: toDocumentSegment(nameNode),
        local: true
      };
    }
    return void 0;
  }
};

// node_modules/langium/lib/utils/collections.js
var MultiMap = class {
  constructor(elements) {
    this.map = /* @__PURE__ */ new Map();
    if (elements) {
      for (const [key, value] of elements) {
        this.add(key, value);
      }
    }
  }
  /**
   * The total number of values in the multimap.
   */
  get size() {
    return Reduction.sum(stream(this.map.values()).map((a) => a.length));
  }
  /**
   * Clear all entries in the multimap.
   */
  clear() {
    this.map.clear();
  }
  /**
   * Operates differently depending on whether a `value` is given:
   *  * With a value, this method deletes the specific key / value pair from the multimap.
   *  * Without a value, all values associated with the given key are deleted.
   *
   * @returns `true` if a value existed and has been removed, or `false` if the specified
   *     key / value does not exist.
   */
  delete(key, value) {
    if (value === void 0) {
      return this.map.delete(key);
    } else {
      const values2 = this.map.get(key);
      if (values2) {
        const index = values2.indexOf(value);
        if (index >= 0) {
          if (values2.length === 1) {
            this.map.delete(key);
          } else {
            values2.splice(index, 1);
          }
          return true;
        }
      }
      return false;
    }
  }
  /**
   * Returns an array of all values associated with the given key. If no value exists,
   * an empty array is returned.
   *
   * _Note:_ The returned array is assumed not to be modified. Use the `set` method to add a
   * value and `delete` to remove a value from the multimap.
   */
  get(key) {
    var _a;
    return (_a = this.map.get(key)) !== null && _a !== void 0 ? _a : [];
  }
  /**
   * Operates differently depending on whether a `value` is given:
   *  * With a value, this method returns `true` if the specific key / value pair is present in the multimap.
   *  * Without a value, this method returns `true` if the given key is present in the multimap.
   */
  has(key, value) {
    if (value === void 0) {
      return this.map.has(key);
    } else {
      const values2 = this.map.get(key);
      if (values2) {
        return values2.indexOf(value) >= 0;
      }
      return false;
    }
  }
  /**
   * Add the given key / value pair to the multimap.
   */
  add(key, value) {
    if (this.map.has(key)) {
      this.map.get(key).push(value);
    } else {
      this.map.set(key, [value]);
    }
    return this;
  }
  /**
   * Add the given set of key / value pairs to the multimap.
   */
  addAll(key, values2) {
    if (this.map.has(key)) {
      this.map.get(key).push(...values2);
    } else {
      this.map.set(key, Array.from(values2));
    }
    return this;
  }
  /**
   * Invokes the given callback function for every key / value pair in the multimap.
   */
  forEach(callbackfn) {
    this.map.forEach((array, key) => array.forEach((value) => callbackfn(value, key, this)));
  }
  /**
   * Returns an iterator of key, value pairs for every entry in the map.
   */
  [Symbol.iterator]() {
    return this.entries().iterator();
  }
  /**
   * Returns a stream of key, value pairs for every entry in the map.
   */
  entries() {
    return stream(this.map.entries()).flatMap(([key, array]) => array.map((value) => [key, value]));
  }
  /**
   * Returns a stream of keys in the map.
   */
  keys() {
    return stream(this.map.keys());
  }
  /**
   * Returns a stream of values in the map.
   */
  values() {
    return stream(this.map.values()).flat();
  }
  /**
   * Returns a stream of key, value set pairs for every key in the map.
   */
  entriesGroupedByKey() {
    return stream(this.map.entries());
  }
};
var BiMap = class {
  get size() {
    return this.map.size;
  }
  constructor(elements) {
    this.map = /* @__PURE__ */ new Map();
    this.inverse = /* @__PURE__ */ new Map();
    if (elements) {
      for (const [key, value] of elements) {
        this.set(key, value);
      }
    }
  }
  clear() {
    this.map.clear();
    this.inverse.clear();
  }
  set(key, value) {
    this.map.set(key, value);
    this.inverse.set(value, key);
    return this;
  }
  get(key) {
    return this.map.get(key);
  }
  getKey(value) {
    return this.inverse.get(value);
  }
  delete(key) {
    const value = this.map.get(key);
    if (value !== void 0) {
      this.map.delete(key);
      this.inverse.delete(value);
      return true;
    }
    return false;
  }
};

// node_modules/langium/lib/references/scope-computation.js
var DefaultScopeComputation = class {
  constructor(services) {
    this.nameProvider = services.references.NameProvider;
    this.descriptions = services.workspace.AstNodeDescriptionProvider;
  }
  async computeExports(document, cancelToken = cancellation_exports.CancellationToken.None) {
    return this.computeExportsForNode(document.parseResult.value, document, void 0, cancelToken);
  }
  /**
   * Creates {@link AstNodeDescription AstNodeDescriptions} for the given {@link AstNode parentNode} and its children.
   * The list of children to be considered is determined by the function parameter {@link children}.
   * By default only the direct children of {@link parentNode} are visited, nested nodes are not exported.
   *
   * @param parentNode AST node to be exported, i.e., of which an {@link AstNodeDescription} shall be added to the returned list.
   * @param document The document containing the AST node to be exported.
   * @param children A function called with {@link parentNode} as single argument and returning an {@link Iterable} supplying the children to be visited, which must be directly or transitively contained in {@link parentNode}.
   * @param cancelToken Indicates when to cancel the current operation.
   * @throws `OperationCancelled` if a user action occurs during execution.
   * @returns A list of {@link AstNodeDescription AstNodeDescriptions} to be published to index.
   */
  async computeExportsForNode(parentNode, document, children = streamContents, cancelToken = cancellation_exports.CancellationToken.None) {
    const exports2 = [];
    this.exportNode(parentNode, exports2, document);
    for (const node of children(parentNode)) {
      await interruptAndCheck(cancelToken);
      this.exportNode(node, exports2, document);
    }
    return exports2;
  }
  /**
   * Add a single node to the list of exports if it has a name. Override this method to change how
   * symbols are exported, e.g. by modifying their exported name.
   */
  exportNode(node, exports2, document) {
    const name = this.nameProvider.getName(node);
    if (name) {
      exports2.push(this.descriptions.createDescription(node, name, document));
    }
  }
  async computeLocalScopes(document, cancelToken = cancellation_exports.CancellationToken.None) {
    const rootNode = document.parseResult.value;
    const scopes = new MultiMap();
    for (const node of streamAllContents(rootNode)) {
      await interruptAndCheck(cancelToken);
      this.processNode(node, document, scopes);
    }
    return scopes;
  }
  /**
   * Process a single node during scopes computation. The default implementation makes the node visible
   * in the subtree of its container (if the node has a name). Override this method to change this,
   * e.g. by increasing the visibility to a higher level in the AST.
   */
  processNode(node, document, scopes) {
    const container = node.$container;
    if (container) {
      const name = this.nameProvider.getName(node);
      if (name) {
        scopes.add(container, this.descriptions.createDescription(node, name, document));
      }
    }
  }
};

// node_modules/langium/lib/references/scope.js
var StreamScope = class {
  constructor(elements, outerScope, options) {
    var _a;
    this.elements = elements;
    this.outerScope = outerScope;
    this.caseInsensitive = (_a = options === null || options === void 0 ? void 0 : options.caseInsensitive) !== null && _a !== void 0 ? _a : false;
  }
  getAllElements() {
    if (this.outerScope) {
      return this.elements.concat(this.outerScope.getAllElements());
    } else {
      return this.elements;
    }
  }
  getElement(name) {
    const local = this.caseInsensitive ? this.elements.find((e) => e.name.toLowerCase() === name.toLowerCase()) : this.elements.find((e) => e.name === name);
    if (local) {
      return local;
    }
    if (this.outerScope) {
      return this.outerScope.getElement(name);
    }
    return void 0;
  }
};
var MapScope = class {
  constructor(elements, outerScope, options) {
    var _a;
    this.elements = /* @__PURE__ */ new Map();
    this.caseInsensitive = (_a = options === null || options === void 0 ? void 0 : options.caseInsensitive) !== null && _a !== void 0 ? _a : false;
    for (const element of elements) {
      const name = this.caseInsensitive ? element.name.toLowerCase() : element.name;
      this.elements.set(name, element);
    }
    this.outerScope = outerScope;
  }
  getElement(name) {
    const localName = this.caseInsensitive ? name.toLowerCase() : name;
    const local = this.elements.get(localName);
    if (local) {
      return local;
    }
    if (this.outerScope) {
      return this.outerScope.getElement(name);
    }
    return void 0;
  }
  getAllElements() {
    let elementStream = stream(this.elements.values());
    if (this.outerScope) {
      elementStream = elementStream.concat(this.outerScope.getAllElements());
    }
    return elementStream;
  }
};
var EMPTY_SCOPE = {
  getElement() {
    return void 0;
  },
  getAllElements() {
    return EMPTY_STREAM;
  }
};

// node_modules/langium/lib/utils/caching.js
var DisposableCache = class {
  constructor() {
    this.toDispose = [];
    this.isDisposed = false;
  }
  onDispose(disposable) {
    this.toDispose.push(disposable);
  }
  dispose() {
    this.throwIfDisposed();
    this.clear();
    this.isDisposed = true;
    this.toDispose.forEach((disposable) => disposable.dispose());
  }
  throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error("This cache has already been disposed");
    }
  }
};
var SimpleCache = class extends DisposableCache {
  constructor() {
    super(...arguments);
    this.cache = /* @__PURE__ */ new Map();
  }
  has(key) {
    this.throwIfDisposed();
    return this.cache.has(key);
  }
  set(key, value) {
    this.throwIfDisposed();
    this.cache.set(key, value);
  }
  get(key, provider) {
    this.throwIfDisposed();
    if (this.cache.has(key)) {
      return this.cache.get(key);
    } else if (provider) {
      const value = provider();
      this.cache.set(key, value);
      return value;
    } else {
      return void 0;
    }
  }
  delete(key) {
    this.throwIfDisposed();
    return this.cache.delete(key);
  }
  clear() {
    this.throwIfDisposed();
    this.cache.clear();
  }
};
var ContextCache = class extends DisposableCache {
  constructor(converter) {
    super();
    this.cache = /* @__PURE__ */ new Map();
    this.converter = converter !== null && converter !== void 0 ? converter : (value) => value;
  }
  has(contextKey, key) {
    this.throwIfDisposed();
    return this.cacheForContext(contextKey).has(key);
  }
  set(contextKey, key, value) {
    this.throwIfDisposed();
    this.cacheForContext(contextKey).set(key, value);
  }
  get(contextKey, key, provider) {
    this.throwIfDisposed();
    const contextCache = this.cacheForContext(contextKey);
    if (contextCache.has(key)) {
      return contextCache.get(key);
    } else if (provider) {
      const value = provider();
      contextCache.set(key, value);
      return value;
    } else {
      return void 0;
    }
  }
  delete(contextKey, key) {
    this.throwIfDisposed();
    return this.cacheForContext(contextKey).delete(key);
  }
  clear(contextKey) {
    this.throwIfDisposed();
    if (contextKey) {
      const mapKey = this.converter(contextKey);
      this.cache.delete(mapKey);
    } else {
      this.cache.clear();
    }
  }
  cacheForContext(contextKey) {
    const mapKey = this.converter(contextKey);
    let documentCache = this.cache.get(mapKey);
    if (!documentCache) {
      documentCache = /* @__PURE__ */ new Map();
      this.cache.set(mapKey, documentCache);
    }
    return documentCache;
  }
};
var DocumentCache = class extends ContextCache {
  /**
   * Creates a new document cache.
   *
   * @param sharedServices Service container instance to hook into document lifecycle events.
   * @param state Optional document state on which the cache should evict.
   * If not provided, the cache will evict on `DocumentBuilder#onUpdate`.
   * *Deleted* documents are considered in both cases.
   *
   * Providing a state here will use `DocumentBuilder#onDocumentPhase` instead,
   * which triggers on all documents that have been affected by this change, assuming that the
   * state is `DocumentState.Linked` or a later state.
   */
  constructor(sharedServices, state) {
    super((uri) => uri.toString());
    if (state) {
      this.toDispose.push(sharedServices.workspace.DocumentBuilder.onDocumentPhase(state, (document) => {
        this.clear(document.uri.toString());
      }));
      this.toDispose.push(sharedServices.workspace.DocumentBuilder.onUpdate((_changed, deleted) => {
        for (const uri of deleted) {
          this.clear(uri);
        }
      }));
    } else {
      this.toDispose.push(sharedServices.workspace.DocumentBuilder.onUpdate((changed, deleted) => {
        const allUris = changed.concat(deleted);
        for (const uri of allUris) {
          this.clear(uri);
        }
      }));
    }
  }
};
var WorkspaceCache = class extends SimpleCache {
  /**
   * Creates a new workspace cache.
   *
   * @param sharedServices Service container instance to hook into document lifecycle events.
   * @param state Optional document state on which the cache should evict.
   * If not provided, the cache will evict on `DocumentBuilder#onUpdate`.
   * *Deleted* documents are considered in both cases.
   */
  constructor(sharedServices, state) {
    super();
    if (state) {
      this.toDispose.push(sharedServices.workspace.DocumentBuilder.onBuildPhase(state, () => {
        this.clear();
      }));
      this.toDispose.push(sharedServices.workspace.DocumentBuilder.onUpdate((_changed, deleted) => {
        if (deleted.length > 0) {
          this.clear();
        }
      }));
    } else {
      this.toDispose.push(sharedServices.workspace.DocumentBuilder.onUpdate(() => {
        this.clear();
      }));
    }
  }
};

// node_modules/langium/lib/references/scope-provider.js
var DefaultScopeProvider = class {
  constructor(services) {
    this.reflection = services.shared.AstReflection;
    this.nameProvider = services.references.NameProvider;
    this.descriptions = services.workspace.AstNodeDescriptionProvider;
    this.indexManager = services.shared.workspace.IndexManager;
    this.globalScopeCache = new WorkspaceCache(services.shared);
  }
  getScope(context) {
    const scopes = [];
    const referenceType = this.reflection.getReferenceType(context);
    const precomputed = getDocument(context.container).precomputedScopes;
    if (precomputed) {
      let currentNode = context.container;
      do {
        const allDescriptions = precomputed.get(currentNode);
        if (allDescriptions.length > 0) {
          scopes.push(stream(allDescriptions).filter((desc) => this.reflection.isSubtype(desc.type, referenceType)));
        }
        currentNode = currentNode.$container;
      } while (currentNode);
    }
    let result2 = this.getGlobalScope(referenceType, context);
    for (let i = scopes.length - 1; i >= 0; i--) {
      result2 = this.createScope(scopes[i], result2);
    }
    return result2;
  }
  /**
   * Create a scope for the given collection of AST node descriptions.
   */
  createScope(elements, outerScope, options) {
    return new StreamScope(stream(elements), outerScope, options);
  }
  /**
   * Create a scope for the given collection of AST nodes, which need to be transformed into respective
   * descriptions first. This is done using the `NameProvider` and `AstNodeDescriptionProvider` services.
   */
  createScopeForNodes(elements, outerScope, options) {
    const s = stream(elements).map((e) => {
      const name = this.nameProvider.getName(e);
      if (name) {
        return this.descriptions.createDescription(e, name);
      }
      return void 0;
    }).nonNullable();
    return new StreamScope(s, outerScope, options);
  }
  /**
   * Create a global scope filtered for the given reference type.
   */
  getGlobalScope(referenceType, _context) {
    return this.globalScopeCache.get(referenceType, () => new MapScope(this.indexManager.allElements(referenceType)));
  }
};

// node_modules/langium/lib/serializer/json-serializer.js
function isAstNodeWithComment(node) {
  return typeof node.$comment === "string";
}
function isIntermediateReference(obj) {
  return typeof obj === "object" && !!obj && ("$ref" in obj || "$error" in obj);
}
var DefaultJsonSerializer = class {
  constructor(services) {
    this.ignoreProperties = /* @__PURE__ */ new Set(["$container", "$containerProperty", "$containerIndex", "$document", "$cstNode"]);
    this.langiumDocuments = services.shared.workspace.LangiumDocuments;
    this.astNodeLocator = services.workspace.AstNodeLocator;
    this.nameProvider = services.references.NameProvider;
    this.commentProvider = services.documentation.CommentProvider;
  }
  serialize(node, options) {
    const serializeOptions = options !== null && options !== void 0 ? options : {};
    const specificReplacer = options === null || options === void 0 ? void 0 : options.replacer;
    const defaultReplacer = (key, value) => this.replacer(key, value, serializeOptions);
    const replacer = specificReplacer ? (key, value) => specificReplacer(key, value, defaultReplacer) : defaultReplacer;
    try {
      this.currentDocument = getDocument(node);
      return JSON.stringify(node, replacer, options === null || options === void 0 ? void 0 : options.space);
    } finally {
      this.currentDocument = void 0;
    }
  }
  deserialize(content, options) {
    const deserializeOptions = options !== null && options !== void 0 ? options : {};
    const root2 = JSON.parse(content);
    this.linkNode(root2, root2, deserializeOptions);
    return root2;
  }
  replacer(key, value, { refText, sourceText, textRegions, comments, uriConverter }) {
    var _a, _b, _c, _d;
    if (this.ignoreProperties.has(key)) {
      return void 0;
    } else if (isReference(value)) {
      const refValue = value.ref;
      const $refText = refText ? value.$refText : void 0;
      if (refValue) {
        const targetDocument = getDocument(refValue);
        let targetUri = "";
        if (this.currentDocument && this.currentDocument !== targetDocument) {
          if (uriConverter) {
            targetUri = uriConverter(targetDocument.uri, value);
          } else {
            targetUri = targetDocument.uri.toString();
          }
        }
        const targetPath = this.astNodeLocator.getAstNodePath(refValue);
        return {
          $ref: `${targetUri}#${targetPath}`,
          $refText
        };
      } else {
        return {
          $error: (_b = (_a = value.error) === null || _a === void 0 ? void 0 : _a.message) !== null && _b !== void 0 ? _b : "Could not resolve reference",
          $refText
        };
      }
    } else if (isAstNode(value)) {
      let astNode = void 0;
      if (textRegions) {
        astNode = this.addAstNodeRegionWithAssignmentsTo(Object.assign({}, value));
        if ((!key || value.$document) && (astNode === null || astNode === void 0 ? void 0 : astNode.$textRegion)) {
          astNode.$textRegion.documentURI = (_c = this.currentDocument) === null || _c === void 0 ? void 0 : _c.uri.toString();
        }
      }
      if (sourceText && !key) {
        astNode !== null && astNode !== void 0 ? astNode : astNode = Object.assign({}, value);
        astNode.$sourceText = (_d = value.$cstNode) === null || _d === void 0 ? void 0 : _d.text;
      }
      if (comments) {
        astNode !== null && astNode !== void 0 ? astNode : astNode = Object.assign({}, value);
        const comment = this.commentProvider.getComment(value);
        if (comment) {
          astNode.$comment = comment.replace(/\r/g, "");
        }
      }
      return astNode !== null && astNode !== void 0 ? astNode : value;
    } else {
      return value;
    }
  }
  addAstNodeRegionWithAssignmentsTo(node) {
    const createDocumentSegment = (cstNode) => ({
      offset: cstNode.offset,
      end: cstNode.end,
      length: cstNode.length,
      range: cstNode.range
    });
    if (node.$cstNode) {
      const textRegion = node.$textRegion = createDocumentSegment(node.$cstNode);
      const assignments = textRegion.assignments = {};
      Object.keys(node).filter((key) => !key.startsWith("$")).forEach((key) => {
        const propertyAssignments = findNodesForProperty(node.$cstNode, key).map(createDocumentSegment);
        if (propertyAssignments.length !== 0) {
          assignments[key] = propertyAssignments;
        }
      });
      return node;
    }
    return void 0;
  }
  linkNode(node, root2, options, container, containerProperty, containerIndex) {
    for (const [propertyName, item] of Object.entries(node)) {
      if (Array.isArray(item)) {
        for (let index = 0; index < item.length; index++) {
          const element = item[index];
          if (isIntermediateReference(element)) {
            item[index] = this.reviveReference(node, propertyName, root2, element, options);
          } else if (isAstNode(element)) {
            this.linkNode(element, root2, options, node, propertyName, index);
          }
        }
      } else if (isIntermediateReference(item)) {
        node[propertyName] = this.reviveReference(node, propertyName, root2, item, options);
      } else if (isAstNode(item)) {
        this.linkNode(item, root2, options, node, propertyName);
      }
    }
    const mutable = node;
    mutable.$container = container;
    mutable.$containerProperty = containerProperty;
    mutable.$containerIndex = containerIndex;
  }
  reviveReference(container, property2, root2, reference, options) {
    let refText = reference.$refText;
    let error = reference.$error;
    if (reference.$ref) {
      const ref = this.getRefNode(root2, reference.$ref, options.uriConverter);
      if (isAstNode(ref)) {
        if (!refText) {
          refText = this.nameProvider.getName(ref);
        }
        return {
          $refText: refText !== null && refText !== void 0 ? refText : "",
          ref
        };
      } else {
        error = ref;
      }
    }
    if (error) {
      const ref = {
        $refText: refText !== null && refText !== void 0 ? refText : ""
      };
      ref.error = {
        container,
        property: property2,
        message: error,
        reference: ref
      };
      return ref;
    } else {
      return void 0;
    }
  }
  getRefNode(root2, uri, uriConverter) {
    try {
      const fragmentIndex = uri.indexOf("#");
      if (fragmentIndex === 0) {
        const node2 = this.astNodeLocator.getAstNode(root2, uri.substring(1));
        if (!node2) {
          return "Could not resolve path: " + uri;
        }
        return node2;
      }
      if (fragmentIndex < 0) {
        const documentUri2 = uriConverter ? uriConverter(uri) : URI2.parse(uri);
        const document2 = this.langiumDocuments.getDocument(documentUri2);
        if (!document2) {
          return "Could not find document for URI: " + uri;
        }
        return document2.parseResult.value;
      }
      const documentUri = uriConverter ? uriConverter(uri.substring(0, fragmentIndex)) : URI2.parse(uri.substring(0, fragmentIndex));
      const document = this.langiumDocuments.getDocument(documentUri);
      if (!document) {
        return "Could not find document for URI: " + uri;
      }
      if (fragmentIndex === uri.length - 1) {
        return document.parseResult.value;
      }
      const node = this.astNodeLocator.getAstNode(document.parseResult.value, uri.substring(fragmentIndex + 1));
      if (!node) {
        return "Could not resolve URI: " + uri;
      }
      return node;
    } catch (err) {
      return String(err);
    }
  }
};

// node_modules/langium/lib/service-registry.js
var DefaultServiceRegistry = class {
  /**
   * @deprecated Use the new `fileExtensionMap` (or `languageIdMap`) property instead.
   */
  get map() {
    return this.fileExtensionMap;
  }
  constructor(services) {
    this.languageIdMap = /* @__PURE__ */ new Map();
    this.fileExtensionMap = /* @__PURE__ */ new Map();
    this.textDocuments = services === null || services === void 0 ? void 0 : services.workspace.TextDocuments;
  }
  register(language) {
    const data = language.LanguageMetaData;
    for (const ext of data.fileExtensions) {
      if (this.fileExtensionMap.has(ext)) {
        console.warn(`The file extension ${ext} is used by multiple languages. It is now assigned to '${data.languageId}'.`);
      }
      this.fileExtensionMap.set(ext, language);
    }
    this.languageIdMap.set(data.languageId, language);
    if (this.languageIdMap.size === 1) {
      this.singleton = language;
    } else {
      this.singleton = void 0;
    }
  }
  getServices(uri) {
    var _a, _b;
    if (this.singleton !== void 0) {
      return this.singleton;
    }
    if (this.languageIdMap.size === 0) {
      throw new Error("The service registry is empty. Use `register` to register the services of a language.");
    }
    const languageId = (_b = (_a = this.textDocuments) === null || _a === void 0 ? void 0 : _a.get(uri)) === null || _b === void 0 ? void 0 : _b.languageId;
    if (languageId !== void 0) {
      const services2 = this.languageIdMap.get(languageId);
      if (services2) {
        return services2;
      }
    }
    const ext = UriUtils.extname(uri);
    const services = this.fileExtensionMap.get(ext);
    if (!services) {
      if (languageId) {
        throw new Error(`The service registry contains no services for the extension '${ext}' for language '${languageId}'.`);
      } else {
        throw new Error(`The service registry contains no services for the extension '${ext}'.`);
      }
    }
    return services;
  }
  hasServices(uri) {
    try {
      this.getServices(uri);
      return true;
    } catch (_a) {
      return false;
    }
  }
  get all() {
    return Array.from(this.languageIdMap.values());
  }
};

// node_modules/langium/lib/validation/validation-registry.js
function diagnosticData(code) {
  return { code };
}
var ValidationCategory;
(function(ValidationCategory2) {
  ValidationCategory2.all = ["fast", "slow", "built-in"];
})(ValidationCategory || (ValidationCategory = {}));
var ValidationRegistry = class {
  constructor(services) {
    this.entries = new MultiMap();
    this.entriesBefore = [];
    this.entriesAfter = [];
    this.reflection = services.shared.AstReflection;
  }
  /**
   * Register a set of validation checks. Each value in the record can be either a single validation check (i.e. a function)
   * or an array of validation checks.
   *
   * @param checksRecord Set of validation checks to register.
   * @param category Optional category for the validation checks (defaults to `'fast'`).
   * @param thisObj Optional object to be used as `this` when calling the validation check functions.
   */
  register(checksRecord, thisObj = this, category = "fast") {
    if (category === "built-in") {
      throw new Error("The 'built-in' category is reserved for lexer, parser, and linker errors.");
    }
    for (const [type, ch] of Object.entries(checksRecord)) {
      const callbacks = ch;
      if (Array.isArray(callbacks)) {
        for (const check of callbacks) {
          const entry = {
            check: this.wrapValidationException(check, thisObj),
            category
          };
          this.addEntry(type, entry);
        }
      } else if (typeof callbacks === "function") {
        const entry = {
          check: this.wrapValidationException(callbacks, thisObj),
          category
        };
        this.addEntry(type, entry);
      } else {
        assertUnreachable(callbacks);
      }
    }
  }
  wrapValidationException(check, thisObj) {
    return async (node, accept, cancelToken) => {
      await this.handleException(() => check.call(thisObj, node, accept, cancelToken), "An error occurred during validation", accept, node);
    };
  }
  async handleException(functionality, messageContext, accept, node) {
    try {
      await functionality();
    } catch (err) {
      if (isOperationCancelled(err)) {
        throw err;
      }
      console.error(`${messageContext}:`, err);
      if (err instanceof Error && err.stack) {
        console.error(err.stack);
      }
      const messageDetails = err instanceof Error ? err.message : String(err);
      accept("error", `${messageContext}: ${messageDetails}`, { node });
    }
  }
  addEntry(type, entry) {
    if (type === "AstNode") {
      this.entries.add("AstNode", entry);
      return;
    }
    for (const subtype of this.reflection.getAllSubTypes(type)) {
      this.entries.add(subtype, entry);
    }
  }
  getChecks(type, categories) {
    let checks = stream(this.entries.get(type)).concat(this.entries.get("AstNode"));
    if (categories) {
      checks = checks.filter((entry) => categories.includes(entry.category));
    }
    return checks.map((entry) => entry.check);
  }
  /**
   * Register logic which will be executed once before validating all the nodes of an AST/Langium document.
   * This helps to prepare or initialize some information which are required or reusable for the following checks on the AstNodes.
   *
   * As an example, for validating unique fully-qualified names of nodes in the AST,
   * here the map for mapping names to nodes could be established.
   * During the usual checks on the nodes, they are put into this map with their name.
   *
   * Note that this approach makes validations stateful, which is relevant e.g. when cancelling the validation.
   * Therefore it is recommended to clear stored information
   * _before_ validating an AST to validate each AST unaffected from other ASTs
   * AND _after_ validating the AST to free memory by information which are no longer used.
   *
   * @param checkBefore a set-up function which will be called once before actually validating an AST
   * @param thisObj Optional object to be used as `this` when calling the validation check functions.
   */
  registerBeforeDocument(checkBefore, thisObj = this) {
    this.entriesBefore.push(this.wrapPreparationException(checkBefore, "An error occurred during set-up of the validation", thisObj));
  }
  /**
   * Register logic which will be executed once after validating all the nodes of an AST/Langium document.
   * This helps to finally evaluate information which are collected during the checks on the AstNodes.
   *
   * As an example, for validating unique fully-qualified names of nodes in the AST,
   * here the map with all the collected nodes and their names is checked
   * and validation hints are created for all nodes with the same name.
   *
   * Note that this approach makes validations stateful, which is relevant e.g. when cancelling the validation.
   * Therefore it is recommended to clear stored information
   * _before_ validating an AST to validate each AST unaffected from other ASTs
   * AND _after_ validating the AST to free memory by information which are no longer used.
   *
   * @param checkBefore a set-up function which will be called once before actually validating an AST
   * @param thisObj Optional object to be used as `this` when calling the validation check functions.
   */
  registerAfterDocument(checkAfter, thisObj = this) {
    this.entriesAfter.push(this.wrapPreparationException(checkAfter, "An error occurred during tear-down of the validation", thisObj));
  }
  wrapPreparationException(check, messageContext, thisObj) {
    return async (rootNode, accept, categories, cancelToken) => {
      await this.handleException(() => check.call(thisObj, rootNode, accept, categories, cancelToken), messageContext, accept, rootNode);
    };
  }
  get checksBefore() {
    return this.entriesBefore;
  }
  get checksAfter() {
    return this.entriesAfter;
  }
};

// node_modules/langium/lib/validation/document-validator.js
var DefaultDocumentValidator = class {
  constructor(services) {
    this.validationRegistry = services.validation.ValidationRegistry;
    this.metadata = services.LanguageMetaData;
  }
  async validateDocument(document, options = {}, cancelToken = cancellation_exports.CancellationToken.None) {
    const parseResult = document.parseResult;
    const diagnostics = [];
    await interruptAndCheck(cancelToken);
    if (!options.categories || options.categories.includes("built-in")) {
      this.processLexingErrors(parseResult, diagnostics, options);
      if (options.stopAfterLexingErrors && diagnostics.some((d) => {
        var _a;
        return ((_a = d.data) === null || _a === void 0 ? void 0 : _a.code) === DocumentValidator.LexingError;
      })) {
        return diagnostics;
      }
      this.processParsingErrors(parseResult, diagnostics, options);
      if (options.stopAfterParsingErrors && diagnostics.some((d) => {
        var _a;
        return ((_a = d.data) === null || _a === void 0 ? void 0 : _a.code) === DocumentValidator.ParsingError;
      })) {
        return diagnostics;
      }
      this.processLinkingErrors(document, diagnostics, options);
      if (options.stopAfterLinkingErrors && diagnostics.some((d) => {
        var _a;
        return ((_a = d.data) === null || _a === void 0 ? void 0 : _a.code) === DocumentValidator.LinkingError;
      })) {
        return diagnostics;
      }
    }
    try {
      diagnostics.push(...await this.validateAst(parseResult.value, options, cancelToken));
    } catch (err) {
      if (isOperationCancelled(err)) {
        throw err;
      }
      console.error("An error occurred during validation:", err);
    }
    await interruptAndCheck(cancelToken);
    return diagnostics;
  }
  processLexingErrors(parseResult, diagnostics, _options) {
    var _a, _b, _c;
    const lexerDiagnostics = [...parseResult.lexerErrors, ...(_b = (_a = parseResult.lexerReport) === null || _a === void 0 ? void 0 : _a.diagnostics) !== null && _b !== void 0 ? _b : []];
    for (const lexerDiagnostic of lexerDiagnostics) {
      const severity = (_c = lexerDiagnostic.severity) !== null && _c !== void 0 ? _c : "error";
      const diagnostic = {
        severity: toDiagnosticSeverity(severity),
        range: {
          start: {
            line: lexerDiagnostic.line - 1,
            character: lexerDiagnostic.column - 1
          },
          end: {
            line: lexerDiagnostic.line - 1,
            character: lexerDiagnostic.column + lexerDiagnostic.length - 1
          }
        },
        message: lexerDiagnostic.message,
        data: toDiagnosticData(severity),
        source: this.getSource()
      };
      diagnostics.push(diagnostic);
    }
  }
  processParsingErrors(parseResult, diagnostics, _options) {
    for (const parserError of parseResult.parserErrors) {
      let range2 = void 0;
      if (isNaN(parserError.token.startOffset)) {
        if ("previousToken" in parserError) {
          const token = parserError.previousToken;
          if (!isNaN(token.startOffset)) {
            const position = { line: token.endLine - 1, character: token.endColumn };
            range2 = { start: position, end: position };
          } else {
            const position = { line: 0, character: 0 };
            range2 = { start: position, end: position };
          }
        }
      } else {
        range2 = tokenToRange(parserError.token);
      }
      if (range2) {
        const diagnostic = {
          severity: toDiagnosticSeverity("error"),
          range: range2,
          message: parserError.message,
          data: diagnosticData(DocumentValidator.ParsingError),
          source: this.getSource()
        };
        diagnostics.push(diagnostic);
      }
    }
  }
  processLinkingErrors(document, diagnostics, _options) {
    for (const reference of document.references) {
      const linkingError = reference.error;
      if (linkingError) {
        const info = {
          node: linkingError.container,
          property: linkingError.property,
          index: linkingError.index,
          data: {
            code: DocumentValidator.LinkingError,
            containerType: linkingError.container.$type,
            property: linkingError.property,
            refText: linkingError.reference.$refText
          }
        };
        diagnostics.push(this.toDiagnostic("error", linkingError.message, info));
      }
    }
  }
  async validateAst(rootNode, options, cancelToken = cancellation_exports.CancellationToken.None) {
    const validationItems = [];
    const acceptor = (severity, message, info) => {
      validationItems.push(this.toDiagnostic(severity, message, info));
    };
    await this.validateAstBefore(rootNode, options, acceptor, cancelToken);
    await this.validateAstNodes(rootNode, options, acceptor, cancelToken);
    await this.validateAstAfter(rootNode, options, acceptor, cancelToken);
    return validationItems;
  }
  async validateAstBefore(rootNode, options, acceptor, cancelToken = cancellation_exports.CancellationToken.None) {
    var _a;
    const checksBefore = this.validationRegistry.checksBefore;
    for (const checkBefore of checksBefore) {
      await interruptAndCheck(cancelToken);
      await checkBefore(rootNode, acceptor, (_a = options.categories) !== null && _a !== void 0 ? _a : [], cancelToken);
    }
  }
  async validateAstNodes(rootNode, options, acceptor, cancelToken = cancellation_exports.CancellationToken.None) {
    await Promise.all(streamAst(rootNode).map(async (node) => {
      await interruptAndCheck(cancelToken);
      const checks = this.validationRegistry.getChecks(node.$type, options.categories);
      for (const check of checks) {
        await check(node, acceptor, cancelToken);
      }
    }));
  }
  async validateAstAfter(rootNode, options, acceptor, cancelToken = cancellation_exports.CancellationToken.None) {
    var _a;
    const checksAfter = this.validationRegistry.checksAfter;
    for (const checkAfter of checksAfter) {
      await interruptAndCheck(cancelToken);
      await checkAfter(rootNode, acceptor, (_a = options.categories) !== null && _a !== void 0 ? _a : [], cancelToken);
    }
  }
  toDiagnostic(severity, message, info) {
    return {
      message,
      range: getDiagnosticRange(info),
      severity: toDiagnosticSeverity(severity),
      code: info.code,
      codeDescription: info.codeDescription,
      tags: info.tags,
      relatedInformation: info.relatedInformation,
      data: info.data,
      source: this.getSource()
    };
  }
  getSource() {
    return this.metadata.languageId;
  }
};
function getDiagnosticRange(info) {
  if (info.range) {
    return info.range;
  }
  let cstNode;
  if (typeof info.property === "string") {
    cstNode = findNodeForProperty(info.node.$cstNode, info.property, info.index);
  } else if (typeof info.keyword === "string") {
    cstNode = findNodeForKeyword(info.node.$cstNode, info.keyword, info.index);
  }
  cstNode !== null && cstNode !== void 0 ? cstNode : cstNode = info.node.$cstNode;
  if (!cstNode) {
    return {
      start: { line: 0, character: 0 },
      end: { line: 0, character: 0 }
    };
  }
  return cstNode.range;
}
function toDiagnosticSeverity(severity) {
  switch (severity) {
    case "error":
      return 1;
    case "warning":
      return 2;
    case "info":
      return 3;
    case "hint":
      return 4;
    default:
      throw new Error("Invalid diagnostic severity: " + severity);
  }
}
function toDiagnosticData(severity) {
  switch (severity) {
    case "error":
      return diagnosticData(DocumentValidator.LexingError);
    case "warning":
      return diagnosticData(DocumentValidator.LexingWarning);
    case "info":
      return diagnosticData(DocumentValidator.LexingInfo);
    case "hint":
      return diagnosticData(DocumentValidator.LexingHint);
    default:
      throw new Error("Invalid diagnostic severity: " + severity);
  }
}
var DocumentValidator;
(function(DocumentValidator2) {
  DocumentValidator2.LexingError = "lexing-error";
  DocumentValidator2.LexingWarning = "lexing-warning";
  DocumentValidator2.LexingInfo = "lexing-info";
  DocumentValidator2.LexingHint = "lexing-hint";
  DocumentValidator2.ParsingError = "parsing-error";
  DocumentValidator2.LinkingError = "linking-error";
})(DocumentValidator || (DocumentValidator = {}));

// node_modules/langium/lib/workspace/ast-descriptions.js
var DefaultAstNodeDescriptionProvider = class {
  constructor(services) {
    this.astNodeLocator = services.workspace.AstNodeLocator;
    this.nameProvider = services.references.NameProvider;
  }
  createDescription(node, name, document) {
    const doc = document !== null && document !== void 0 ? document : getDocument(node);
    name !== null && name !== void 0 ? name : name = this.nameProvider.getName(node);
    const path = this.astNodeLocator.getAstNodePath(node);
    if (!name) {
      throw new Error(`Node at path ${path} has no name.`);
    }
    let nameNodeSegment;
    const nameSegmentGetter = () => {
      var _a;
      return nameNodeSegment !== null && nameNodeSegment !== void 0 ? nameNodeSegment : nameNodeSegment = toDocumentSegment((_a = this.nameProvider.getNameNode(node)) !== null && _a !== void 0 ? _a : node.$cstNode);
    };
    return {
      node,
      name,
      get nameSegment() {
        return nameSegmentGetter();
      },
      selectionSegment: toDocumentSegment(node.$cstNode),
      type: node.$type,
      documentUri: doc.uri,
      path
    };
  }
};
var DefaultReferenceDescriptionProvider = class {
  constructor(services) {
    this.nodeLocator = services.workspace.AstNodeLocator;
  }
  async createDescriptions(document, cancelToken = cancellation_exports.CancellationToken.None) {
    const descr = [];
    const rootNode = document.parseResult.value;
    for (const astNode of streamAst(rootNode)) {
      await interruptAndCheck(cancelToken);
      streamReferences(astNode).filter((refInfo) => !isLinkingError(refInfo)).forEach((refInfo) => {
        const description = this.createDescription(refInfo);
        if (description) {
          descr.push(description);
        }
      });
    }
    return descr;
  }
  createDescription(refInfo) {
    const targetNodeDescr = refInfo.reference.$nodeDescription;
    const refCstNode = refInfo.reference.$refNode;
    if (!targetNodeDescr || !refCstNode) {
      return void 0;
    }
    const docUri = getDocument(refInfo.container).uri;
    return {
      sourceUri: docUri,
      sourcePath: this.nodeLocator.getAstNodePath(refInfo.container),
      targetUri: targetNodeDescr.documentUri,
      targetPath: targetNodeDescr.path,
      segment: toDocumentSegment(refCstNode),
      local: UriUtils.equals(targetNodeDescr.documentUri, docUri)
    };
  }
};

// node_modules/langium/lib/workspace/ast-node-locator.js
var DefaultAstNodeLocator = class {
  constructor() {
    this.segmentSeparator = "/";
    this.indexSeparator = "@";
  }
  getAstNodePath(node) {
    if (node.$container) {
      const containerPath = this.getAstNodePath(node.$container);
      const newSegment = this.getPathSegment(node);
      const nodePath = containerPath + this.segmentSeparator + newSegment;
      return nodePath;
    }
    return "";
  }
  getPathSegment({ $containerProperty, $containerIndex }) {
    if (!$containerProperty) {
      throw new Error("Missing '$containerProperty' in AST node.");
    }
    if ($containerIndex !== void 0) {
      return $containerProperty + this.indexSeparator + $containerIndex;
    }
    return $containerProperty;
  }
  getAstNode(node, path) {
    const segments = path.split(this.segmentSeparator);
    return segments.reduce((previousValue, currentValue) => {
      if (!previousValue || currentValue.length === 0) {
        return previousValue;
      }
      const propertyIndex = currentValue.indexOf(this.indexSeparator);
      if (propertyIndex > 0) {
        const property2 = currentValue.substring(0, propertyIndex);
        const arrayIndex = parseInt(currentValue.substring(propertyIndex + 1));
        const array = previousValue[property2];
        return array === null || array === void 0 ? void 0 : array[arrayIndex];
      }
      return previousValue[currentValue];
    }, node);
  }
};

// node_modules/langium/lib/utils/event.js
var event_exports = {};
__reExport(event_exports, __toESM(require_events(), 1));

// node_modules/langium/lib/workspace/configuration.js
var DefaultConfigurationProvider = class {
  constructor(services) {
    this._ready = new Deferred();
    this.settings = {};
    this.workspaceConfig = false;
    this.onConfigurationSectionUpdateEmitter = new event_exports.Emitter();
    this.serviceRegistry = services.ServiceRegistry;
  }
  get ready() {
    return this._ready.promise;
  }
  initialize(params) {
    var _a, _b;
    this.workspaceConfig = (_b = (_a = params.capabilities.workspace) === null || _a === void 0 ? void 0 : _a.configuration) !== null && _b !== void 0 ? _b : false;
  }
  async initialized(params) {
    if (this.workspaceConfig) {
      if (params.register) {
        const languages = this.serviceRegistry.all;
        params.register({
          // Listen to configuration changes for all languages
          section: languages.map((lang) => this.toSectionName(lang.LanguageMetaData.languageId))
        });
      }
      if (params.fetchConfiguration) {
        const configToUpdate = this.serviceRegistry.all.map((lang) => ({
          // Fetch the configuration changes for all languages
          section: this.toSectionName(lang.LanguageMetaData.languageId)
        }));
        const configs = await params.fetchConfiguration(configToUpdate);
        configToUpdate.forEach((conf, idx) => {
          this.updateSectionConfiguration(conf.section, configs[idx]);
        });
      }
    }
    this._ready.resolve();
  }
  /**
   *  Updates the cached configurations using the `change` notification parameters.
   *
   * @param change The parameters of a change configuration notification.
   * `settings` property of the change object could be expressed as `Record<string, Record<string, any>>`
   */
  updateConfiguration(change) {
    if (!change.settings) {
      return;
    }
    Object.keys(change.settings).forEach((section) => {
      const configuration = change.settings[section];
      this.updateSectionConfiguration(section, configuration);
      this.onConfigurationSectionUpdateEmitter.fire({ section, configuration });
    });
  }
  updateSectionConfiguration(section, configuration) {
    this.settings[section] = configuration;
  }
  /**
  * Returns a configuration value stored for the given language.
  *
  * @param language The language id
  * @param configuration Configuration name
  */
  async getConfiguration(language, configuration) {
    await this.ready;
    const sectionName = this.toSectionName(language);
    if (this.settings[sectionName]) {
      return this.settings[sectionName][configuration];
    }
  }
  toSectionName(languageId) {
    return `${languageId}`;
  }
  get onConfigurationSectionUpdate() {
    return this.onConfigurationSectionUpdateEmitter.event;
  }
};

// node_modules/langium/lib/utils/disposable.js
var Disposable;
(function(Disposable2) {
  function create2(callback) {
    return {
      dispose: async () => await callback()
    };
  }
  Disposable2.create = create2;
})(Disposable || (Disposable = {}));

// node_modules/langium/lib/workspace/document-builder.js
var DefaultDocumentBuilder = class {
  constructor(services) {
    this.updateBuildOptions = {
      // Default: run only the built-in validation checks and those in the _fast_ category (includes those without category)
      validation: {
        categories: ["built-in", "fast"]
      }
    };
    this.updateListeners = [];
    this.buildPhaseListeners = new MultiMap();
    this.documentPhaseListeners = new MultiMap();
    this.buildState = /* @__PURE__ */ new Map();
    this.documentBuildWaiters = /* @__PURE__ */ new Map();
    this.currentState = DocumentState.Changed;
    this.langiumDocuments = services.workspace.LangiumDocuments;
    this.langiumDocumentFactory = services.workspace.LangiumDocumentFactory;
    this.textDocuments = services.workspace.TextDocuments;
    this.indexManager = services.workspace.IndexManager;
    this.serviceRegistry = services.ServiceRegistry;
  }
  async build(documents, options = {}, cancelToken = cancellation_exports.CancellationToken.None) {
    var _a, _b;
    for (const document of documents) {
      const key = document.uri.toString();
      if (document.state === DocumentState.Validated) {
        if (typeof options.validation === "boolean" && options.validation) {
          document.state = DocumentState.IndexedReferences;
          document.diagnostics = void 0;
          this.buildState.delete(key);
        } else if (typeof options.validation === "object") {
          const buildState = this.buildState.get(key);
          const previousCategories = (_a = buildState === null || buildState === void 0 ? void 0 : buildState.result) === null || _a === void 0 ? void 0 : _a.validationChecks;
          if (previousCategories) {
            const newCategories = (_b = options.validation.categories) !== null && _b !== void 0 ? _b : ValidationCategory.all;
            const categories = newCategories.filter((c) => !previousCategories.includes(c));
            if (categories.length > 0) {
              this.buildState.set(key, {
                completed: false,
                options: {
                  validation: Object.assign(Object.assign({}, options.validation), { categories })
                },
                result: buildState.result
              });
              document.state = DocumentState.IndexedReferences;
            }
          }
        }
      } else {
        this.buildState.delete(key);
      }
    }
    this.currentState = DocumentState.Changed;
    await this.emitUpdate(documents.map((e) => e.uri), []);
    await this.buildDocuments(documents, options, cancelToken);
  }
  async update(changed, deleted, cancelToken = cancellation_exports.CancellationToken.None) {
    this.currentState = DocumentState.Changed;
    for (const deletedUri of deleted) {
      this.langiumDocuments.deleteDocument(deletedUri);
      this.buildState.delete(deletedUri.toString());
      this.indexManager.remove(deletedUri);
    }
    for (const changedUri of changed) {
      const invalidated = this.langiumDocuments.invalidateDocument(changedUri);
      if (!invalidated) {
        const newDocument = this.langiumDocumentFactory.fromModel({ $type: "INVALID" }, changedUri);
        newDocument.state = DocumentState.Changed;
        this.langiumDocuments.addDocument(newDocument);
      }
      this.buildState.delete(changedUri.toString());
    }
    const allChangedUris = stream(changed).concat(deleted).map((uri) => uri.toString()).toSet();
    this.langiumDocuments.all.filter((doc) => !allChangedUris.has(doc.uri.toString()) && this.shouldRelink(doc, allChangedUris)).forEach((doc) => {
      const linker = this.serviceRegistry.getServices(doc.uri).references.Linker;
      linker.unlink(doc);
      doc.state = Math.min(doc.state, DocumentState.ComputedScopes);
      doc.diagnostics = void 0;
    });
    await this.emitUpdate(changed, deleted);
    await interruptAndCheck(cancelToken);
    const rebuildDocuments = this.sortDocuments(this.langiumDocuments.all.filter((doc) => {
      var _a;
      return doc.state < DocumentState.Linked || !((_a = this.buildState.get(doc.uri.toString())) === null || _a === void 0 ? void 0 : _a.completed);
    }).toArray());
    await this.buildDocuments(rebuildDocuments, this.updateBuildOptions, cancelToken);
  }
  async emitUpdate(changed, deleted) {
    await Promise.all(this.updateListeners.map((listener) => listener(changed, deleted)));
  }
  /**
   * Sort the given documents by priority. By default, documents with an open text document are prioritized.
   * This is useful to ensure that visible documents show their diagnostics before all other documents.
   *
   * This improves the responsiveness in large workspaces as users usually don't care about diagnostics
   * in files that are currently not opened in the editor.
   */
  sortDocuments(documents) {
    let left = 0;
    let right = documents.length - 1;
    while (left < right) {
      while (left < documents.length && this.hasTextDocument(documents[left])) {
        left++;
      }
      while (right >= 0 && !this.hasTextDocument(documents[right])) {
        right--;
      }
      if (left < right) {
        [documents[left], documents[right]] = [documents[right], documents[left]];
      }
    }
    return documents;
  }
  hasTextDocument(doc) {
    var _a;
    return Boolean((_a = this.textDocuments) === null || _a === void 0 ? void 0 : _a.get(doc.uri));
  }
  /**
   * Check whether the given document should be relinked after changes were found in the given URIs.
   */
  shouldRelink(document, changedUris) {
    if (document.references.some((ref) => ref.error !== void 0)) {
      return true;
    }
    return this.indexManager.isAffected(document, changedUris);
  }
  onUpdate(callback) {
    this.updateListeners.push(callback);
    return Disposable.create(() => {
      const index = this.updateListeners.indexOf(callback);
      if (index >= 0) {
        this.updateListeners.splice(index, 1);
      }
    });
  }
  /**
   * Build the given documents by stepping through all build phases. If a document's state indicates
   * that a certain build phase is already done, the phase is skipped for that document.
   *
   * @param documents The documents to build.
   * @param options the {@link BuildOptions} to use.
   * @param cancelToken A cancellation token that can be used to cancel the build.
   * @returns A promise that resolves when the build is done.
   */
  async buildDocuments(documents, options, cancelToken) {
    this.prepareBuild(documents, options);
    await this.runCancelable(documents, DocumentState.Parsed, cancelToken, (doc) => this.langiumDocumentFactory.update(doc, cancelToken));
    await this.runCancelable(documents, DocumentState.IndexedContent, cancelToken, (doc) => this.indexManager.updateContent(doc, cancelToken));
    await this.runCancelable(documents, DocumentState.ComputedScopes, cancelToken, async (doc) => {
      const scopeComputation = this.serviceRegistry.getServices(doc.uri).references.ScopeComputation;
      doc.precomputedScopes = await scopeComputation.computeLocalScopes(doc, cancelToken);
    });
    await this.runCancelable(documents, DocumentState.Linked, cancelToken, (doc) => {
      const linker = this.serviceRegistry.getServices(doc.uri).references.Linker;
      return linker.link(doc, cancelToken);
    });
    await this.runCancelable(documents, DocumentState.IndexedReferences, cancelToken, (doc) => this.indexManager.updateReferences(doc, cancelToken));
    const toBeValidated = documents.filter((doc) => this.shouldValidate(doc));
    await this.runCancelable(toBeValidated, DocumentState.Validated, cancelToken, (doc) => this.validate(doc, cancelToken));
    for (const doc of documents) {
      const state = this.buildState.get(doc.uri.toString());
      if (state) {
        state.completed = true;
      }
    }
  }
  /**
   * Runs prior to beginning the build process to update the {@link DocumentBuildState} for each document
   *
   * @param documents collection of documents to be built
   * @param options the {@link BuildOptions} to use
   */
  prepareBuild(documents, options) {
    for (const doc of documents) {
      const key = doc.uri.toString();
      const state = this.buildState.get(key);
      if (!state || state.completed) {
        this.buildState.set(key, {
          completed: false,
          options,
          result: state === null || state === void 0 ? void 0 : state.result
        });
      }
    }
  }
  /**
   * Runs a cancelable operation on a set of documents to bring them to a specified {@link DocumentState}.
   *
   * @param documents The array of documents to process.
   * @param targetState The target {@link DocumentState} to bring the documents to.
   * @param cancelToken A token that can be used to cancel the operation.
   * @param callback A function to be called for each document.
   * @returns A promise that resolves when all documents have been processed or the operation is canceled.
   * @throws Will throw `OperationCancelled` if the operation is canceled via a `CancellationToken`.
   */
  async runCancelable(documents, targetState, cancelToken, callback) {
    const filtered = documents.filter((doc) => doc.state < targetState);
    for (const document of filtered) {
      await interruptAndCheck(cancelToken);
      await callback(document);
      document.state = targetState;
      await this.notifyDocumentPhase(document, targetState, cancelToken);
    }
    const targetStateDocs = documents.filter((doc) => doc.state === targetState);
    await this.notifyBuildPhase(targetStateDocs, targetState, cancelToken);
    this.currentState = targetState;
  }
  onBuildPhase(targetState, callback) {
    this.buildPhaseListeners.add(targetState, callback);
    return Disposable.create(() => {
      this.buildPhaseListeners.delete(targetState, callback);
    });
  }
  onDocumentPhase(targetState, callback) {
    this.documentPhaseListeners.add(targetState, callback);
    return Disposable.create(() => {
      this.documentPhaseListeners.delete(targetState, callback);
    });
  }
  waitUntil(state, uriOrToken, cancelToken) {
    let uri = void 0;
    if (uriOrToken && "path" in uriOrToken) {
      uri = uriOrToken;
    } else {
      cancelToken = uriOrToken;
    }
    cancelToken !== null && cancelToken !== void 0 ? cancelToken : cancelToken = cancellation_exports.CancellationToken.None;
    if (uri) {
      const document = this.langiumDocuments.getDocument(uri);
      if (document && document.state > state) {
        return Promise.resolve(uri);
      }
    }
    if (this.currentState >= state) {
      return Promise.resolve(void 0);
    } else if (cancelToken.isCancellationRequested) {
      return Promise.reject(OperationCancelled);
    }
    return new Promise((resolve, reject2) => {
      const buildDisposable = this.onBuildPhase(state, () => {
        buildDisposable.dispose();
        cancelDisposable.dispose();
        if (uri) {
          const document = this.langiumDocuments.getDocument(uri);
          resolve(document === null || document === void 0 ? void 0 : document.uri);
        } else {
          resolve(void 0);
        }
      });
      const cancelDisposable = cancelToken.onCancellationRequested(() => {
        buildDisposable.dispose();
        cancelDisposable.dispose();
        reject2(OperationCancelled);
      });
    });
  }
  async notifyDocumentPhase(document, state, cancelToken) {
    const listeners = this.documentPhaseListeners.get(state);
    const listenersCopy = listeners.slice();
    for (const listener of listenersCopy) {
      try {
        await listener(document, cancelToken);
      } catch (err) {
        if (!isOperationCancelled(err)) {
          throw err;
        }
      }
    }
  }
  async notifyBuildPhase(documents, state, cancelToken) {
    if (documents.length === 0) {
      return;
    }
    const listeners = this.buildPhaseListeners.get(state);
    const listenersCopy = listeners.slice();
    for (const listener of listenersCopy) {
      await interruptAndCheck(cancelToken);
      await listener(documents, cancelToken);
    }
  }
  /**
   * Determine whether the given document should be validated during a build. The default
   * implementation checks the `validation` property of the build options. If it's set to `true`
   * or a `ValidationOptions` object, the document is included in the validation phase.
   */
  shouldValidate(document) {
    return Boolean(this.getBuildOptions(document).validation);
  }
  /**
   * Run validation checks on the given document and store the resulting diagnostics in the document.
   * If the document already contains diagnostics, the new ones are added to the list.
   */
  async validate(document, cancelToken) {
    var _a, _b;
    const validator = this.serviceRegistry.getServices(document.uri).validation.DocumentValidator;
    const validationSetting = this.getBuildOptions(document).validation;
    const options = typeof validationSetting === "object" ? validationSetting : void 0;
    const diagnostics = await validator.validateDocument(document, options, cancelToken);
    if (document.diagnostics) {
      document.diagnostics.push(...diagnostics);
    } else {
      document.diagnostics = diagnostics;
    }
    const state = this.buildState.get(document.uri.toString());
    if (state) {
      (_a = state.result) !== null && _a !== void 0 ? _a : state.result = {};
      const newCategories = (_b = options === null || options === void 0 ? void 0 : options.categories) !== null && _b !== void 0 ? _b : ValidationCategory.all;
      if (state.result.validationChecks) {
        state.result.validationChecks.push(...newCategories);
      } else {
        state.result.validationChecks = [...newCategories];
      }
    }
  }
  getBuildOptions(document) {
    var _a, _b;
    return (_b = (_a = this.buildState.get(document.uri.toString())) === null || _a === void 0 ? void 0 : _a.options) !== null && _b !== void 0 ? _b : {};
  }
};

// node_modules/langium/lib/workspace/index-manager.js
var DefaultIndexManager = class {
  constructor(services) {
    this.symbolIndex = /* @__PURE__ */ new Map();
    this.symbolByTypeIndex = new ContextCache();
    this.referenceIndex = /* @__PURE__ */ new Map();
    this.documents = services.workspace.LangiumDocuments;
    this.serviceRegistry = services.ServiceRegistry;
    this.astReflection = services.AstReflection;
  }
  findAllReferences(targetNode, astNodePath) {
    const targetDocUri = getDocument(targetNode).uri;
    const result2 = [];
    this.referenceIndex.forEach((docRefs) => {
      docRefs.forEach((refDescr) => {
        if (UriUtils.equals(refDescr.targetUri, targetDocUri) && refDescr.targetPath === astNodePath) {
          result2.push(refDescr);
        }
      });
    });
    return stream(result2);
  }
  allElements(nodeType, uris) {
    let documentUris = stream(this.symbolIndex.keys());
    if (uris) {
      documentUris = documentUris.filter((uri) => !uris || uris.has(uri));
    }
    return documentUris.map((uri) => this.getFileDescriptions(uri, nodeType)).flat();
  }
  getFileDescriptions(uri, nodeType) {
    var _a;
    if (!nodeType) {
      return (_a = this.symbolIndex.get(uri)) !== null && _a !== void 0 ? _a : [];
    }
    const descriptions = this.symbolByTypeIndex.get(uri, nodeType, () => {
      var _a2;
      const allFileDescriptions = (_a2 = this.symbolIndex.get(uri)) !== null && _a2 !== void 0 ? _a2 : [];
      return allFileDescriptions.filter((e) => this.astReflection.isSubtype(e.type, nodeType));
    });
    return descriptions;
  }
  remove(uri) {
    const uriString = uri.toString();
    this.symbolIndex.delete(uriString);
    this.symbolByTypeIndex.clear(uriString);
    this.referenceIndex.delete(uriString);
  }
  async updateContent(document, cancelToken = cancellation_exports.CancellationToken.None) {
    const services = this.serviceRegistry.getServices(document.uri);
    const exports2 = await services.references.ScopeComputation.computeExports(document, cancelToken);
    const uri = document.uri.toString();
    this.symbolIndex.set(uri, exports2);
    this.symbolByTypeIndex.clear(uri);
  }
  async updateReferences(document, cancelToken = cancellation_exports.CancellationToken.None) {
    const services = this.serviceRegistry.getServices(document.uri);
    const indexData = await services.workspace.ReferenceDescriptionProvider.createDescriptions(document, cancelToken);
    this.referenceIndex.set(document.uri.toString(), indexData);
  }
  isAffected(document, changedUris) {
    const references = this.referenceIndex.get(document.uri.toString());
    if (!references) {
      return false;
    }
    return references.some((ref) => !ref.local && changedUris.has(ref.targetUri.toString()));
  }
};

// node_modules/langium/lib/workspace/workspace-manager.js
var DefaultWorkspaceManager = class {
  constructor(services) {
    this.initialBuildOptions = {};
    this._ready = new Deferred();
    this.serviceRegistry = services.ServiceRegistry;
    this.langiumDocuments = services.workspace.LangiumDocuments;
    this.documentBuilder = services.workspace.DocumentBuilder;
    this.fileSystemProvider = services.workspace.FileSystemProvider;
    this.mutex = services.workspace.WorkspaceLock;
  }
  get ready() {
    return this._ready.promise;
  }
  get workspaceFolders() {
    return this.folders;
  }
  initialize(params) {
    var _a;
    this.folders = (_a = params.workspaceFolders) !== null && _a !== void 0 ? _a : void 0;
  }
  initialized(_params) {
    return this.mutex.write((token) => {
      var _a;
      return this.initializeWorkspace((_a = this.folders) !== null && _a !== void 0 ? _a : [], token);
    });
  }
  async initializeWorkspace(folders, cancelToken = cancellation_exports.CancellationToken.None) {
    const documents = await this.performStartup(folders);
    await interruptAndCheck(cancelToken);
    await this.documentBuilder.build(documents, this.initialBuildOptions, cancelToken);
  }
  /**
   * Performs the uninterruptable startup sequence of the workspace manager.
   * This methods loads all documents in the workspace and other documents and returns them.
   */
  async performStartup(folders) {
    const fileExtensions = this.serviceRegistry.all.flatMap((e) => e.LanguageMetaData.fileExtensions);
    const documents = [];
    const collector = (document) => {
      documents.push(document);
      if (!this.langiumDocuments.hasDocument(document.uri)) {
        this.langiumDocuments.addDocument(document);
      }
    };
    await this.loadAdditionalDocuments(folders, collector);
    await Promise.all(folders.map((wf) => [wf, this.getRootFolder(wf)]).map(async (entry) => this.traverseFolder(...entry, fileExtensions, collector)));
    this._ready.resolve();
    return documents;
  }
  /**
   * Load all additional documents that shall be visible in the context of the given workspace
   * folders and add them to the collector. This can be used to include built-in libraries of
   * your language, which can be either loaded from provided files or constructed in memory.
   */
  loadAdditionalDocuments(_folders, _collector) {
    return Promise.resolve();
  }
  /**
   * Determine the root folder of the source documents in the given workspace folder.
   * The default implementation returns the URI of the workspace folder, but you can override
   * this to return a subfolder like `src` instead.
   */
  getRootFolder(workspaceFolder) {
    return URI2.parse(workspaceFolder.uri);
  }
  /**
   * Traverse the file system folder identified by the given URI and its subfolders. All
   * contained files that match the file extensions are added to the collector.
   */
  async traverseFolder(workspaceFolder, folderPath, fileExtensions, collector) {
    const content = await this.fileSystemProvider.readDirectory(folderPath);
    await Promise.all(content.map(async (entry) => {
      if (this.includeEntry(workspaceFolder, entry, fileExtensions)) {
        if (entry.isDirectory) {
          await this.traverseFolder(workspaceFolder, entry.uri, fileExtensions, collector);
        } else if (entry.isFile) {
          const document = await this.langiumDocuments.getOrCreateDocument(entry.uri);
          collector(document);
        }
      }
    }));
  }
  /**
   * Determine whether the given folder entry shall be included while indexing the workspace.
   */
  includeEntry(_workspaceFolder, entry, fileExtensions) {
    const name = UriUtils.basename(entry.uri);
    if (name.startsWith(".")) {
      return false;
    }
    if (entry.isDirectory) {
      return name !== "node_modules" && name !== "out";
    } else if (entry.isFile) {
      const extname = UriUtils.extname(entry.uri);
      return fileExtensions.includes(extname);
    }
    return false;
  }
};

// node_modules/langium/lib/parser/lexer.js
var DefaultLexerErrorMessageProvider = class {
  buildUnexpectedCharactersMessage(fullText, startOffset, length, line, column) {
    return defaultLexerErrorProvider.buildUnexpectedCharactersMessage(fullText, startOffset, length, line, column);
  }
  buildUnableToPopLexerModeMessage(token) {
    return defaultLexerErrorProvider.buildUnableToPopLexerModeMessage(token);
  }
};
var DEFAULT_TOKENIZE_OPTIONS = { mode: "full" };
var DefaultLexer = class {
  constructor(services) {
    this.errorMessageProvider = services.parser.LexerErrorMessageProvider;
    this.tokenBuilder = services.parser.TokenBuilder;
    const tokens = this.tokenBuilder.buildTokens(services.Grammar, {
      caseInsensitive: services.LanguageMetaData.caseInsensitive
    });
    this.tokenTypes = this.toTokenTypeDictionary(tokens);
    const lexerTokens = isTokenTypeDictionary(tokens) ? Object.values(tokens) : tokens;
    const production = services.LanguageMetaData.mode === "production";
    this.chevrotainLexer = new Lexer(lexerTokens, {
      positionTracking: "full",
      skipValidations: production,
      errorMessageProvider: this.errorMessageProvider
    });
  }
  get definition() {
    return this.tokenTypes;
  }
  tokenize(text, _options = DEFAULT_TOKENIZE_OPTIONS) {
    var _a, _b, _c;
    const chevrotainResult = this.chevrotainLexer.tokenize(text);
    return {
      tokens: chevrotainResult.tokens,
      errors: chevrotainResult.errors,
      hidden: (_a = chevrotainResult.groups.hidden) !== null && _a !== void 0 ? _a : [],
      report: (_c = (_b = this.tokenBuilder).flushLexingReport) === null || _c === void 0 ? void 0 : _c.call(_b, text)
    };
  }
  toTokenTypeDictionary(buildTokens) {
    if (isTokenTypeDictionary(buildTokens))
      return buildTokens;
    const tokens = isIMultiModeLexerDefinition(buildTokens) ? Object.values(buildTokens.modes).flat() : buildTokens;
    const res = {};
    tokens.forEach((token) => res[token.name] = token);
    return res;
  }
};
function isTokenTypeArray(tokenVocabulary) {
  return Array.isArray(tokenVocabulary) && (tokenVocabulary.length === 0 || "name" in tokenVocabulary[0]);
}
function isIMultiModeLexerDefinition(tokenVocabulary) {
  return tokenVocabulary && "modes" in tokenVocabulary && "defaultMode" in tokenVocabulary;
}
function isTokenTypeDictionary(tokenVocabulary) {
  return !isTokenTypeArray(tokenVocabulary) && !isIMultiModeLexerDefinition(tokenVocabulary);
}

// node_modules/langium/lib/documentation/jsdoc.js
function parseJSDoc(node, start, options) {
  let opts;
  let position;
  if (typeof node === "string") {
    position = start;
    opts = options;
  } else {
    position = node.range.start;
    opts = start;
  }
  if (!position) {
    position = Position.create(0, 0);
  }
  const lines = getLines(node);
  const normalizedOptions = normalizeOptions(opts);
  const tokens = tokenize({
    lines,
    position,
    options: normalizedOptions
  });
  return parseJSDocComment({
    index: 0,
    tokens,
    position
  });
}
function isJSDoc(node, options) {
  const normalizedOptions = normalizeOptions(options);
  const lines = getLines(node);
  if (lines.length === 0) {
    return false;
  }
  const first2 = lines[0];
  const last2 = lines[lines.length - 1];
  const firstRegex = normalizedOptions.start;
  const lastRegex = normalizedOptions.end;
  return Boolean(firstRegex === null || firstRegex === void 0 ? void 0 : firstRegex.exec(first2)) && Boolean(lastRegex === null || lastRegex === void 0 ? void 0 : lastRegex.exec(last2));
}
function getLines(node) {
  let content = "";
  if (typeof node === "string") {
    content = node;
  } else {
    content = node.text;
  }
  const lines = content.split(NEWLINE_REGEXP);
  return lines;
}
var tagRegex = /\s*(@([\p{L}][\p{L}\p{N}]*)?)/uy;
var inlineTagRegex = /\{(@[\p{L}][\p{L}\p{N}]*)(\s*)([^\r\n}]+)?\}/gu;
function tokenize(context) {
  var _a, _b, _c;
  const tokens = [];
  let currentLine = context.position.line;
  let currentCharacter = context.position.character;
  for (let i = 0; i < context.lines.length; i++) {
    const first2 = i === 0;
    const last2 = i === context.lines.length - 1;
    let line = context.lines[i];
    let index = 0;
    if (first2 && context.options.start) {
      const match = (_a = context.options.start) === null || _a === void 0 ? void 0 : _a.exec(line);
      if (match) {
        index = match.index + match[0].length;
      }
    } else {
      const match = (_b = context.options.line) === null || _b === void 0 ? void 0 : _b.exec(line);
      if (match) {
        index = match.index + match[0].length;
      }
    }
    if (last2) {
      const match = (_c = context.options.end) === null || _c === void 0 ? void 0 : _c.exec(line);
      if (match) {
        line = line.substring(0, match.index);
      }
    }
    line = line.substring(0, lastCharacter(line));
    const whitespaceEnd = skipWhitespace(line, index);
    if (whitespaceEnd >= line.length) {
      if (tokens.length > 0) {
        const position = Position.create(currentLine, currentCharacter);
        tokens.push({
          type: "break",
          content: "",
          range: Range.create(position, position)
        });
      }
    } else {
      tagRegex.lastIndex = index;
      const tagMatch = tagRegex.exec(line);
      if (tagMatch) {
        const fullMatch = tagMatch[0];
        const value = tagMatch[1];
        const start = Position.create(currentLine, currentCharacter + index);
        const end = Position.create(currentLine, currentCharacter + index + fullMatch.length);
        tokens.push({
          type: "tag",
          content: value,
          range: Range.create(start, end)
        });
        index += fullMatch.length;
        index = skipWhitespace(line, index);
      }
      if (index < line.length) {
        const rest2 = line.substring(index);
        const inlineTagMatches = Array.from(rest2.matchAll(inlineTagRegex));
        tokens.push(...buildInlineTokens(inlineTagMatches, rest2, currentLine, currentCharacter + index));
      }
    }
    currentLine++;
    currentCharacter = 0;
  }
  if (tokens.length > 0 && tokens[tokens.length - 1].type === "break") {
    return tokens.slice(0, -1);
  }
  return tokens;
}
function buildInlineTokens(tags, line, lineIndex, characterIndex) {
  const tokens = [];
  if (tags.length === 0) {
    const start = Position.create(lineIndex, characterIndex);
    const end = Position.create(lineIndex, characterIndex + line.length);
    tokens.push({
      type: "text",
      content: line,
      range: Range.create(start, end)
    });
  } else {
    let lastIndex = 0;
    for (const match of tags) {
      const matchIndex = match.index;
      const startContent = line.substring(lastIndex, matchIndex);
      if (startContent.length > 0) {
        tokens.push({
          type: "text",
          content: line.substring(lastIndex, matchIndex),
          range: Range.create(Position.create(lineIndex, lastIndex + characterIndex), Position.create(lineIndex, matchIndex + characterIndex))
        });
      }
      let offset = startContent.length + 1;
      const tagName = match[1];
      tokens.push({
        type: "inline-tag",
        content: tagName,
        range: Range.create(Position.create(lineIndex, lastIndex + offset + characterIndex), Position.create(lineIndex, lastIndex + offset + tagName.length + characterIndex))
      });
      offset += tagName.length;
      if (match.length === 4) {
        offset += match[2].length;
        const value = match[3];
        tokens.push({
          type: "text",
          content: value,
          range: Range.create(Position.create(lineIndex, lastIndex + offset + characterIndex), Position.create(lineIndex, lastIndex + offset + value.length + characterIndex))
        });
      } else {
        tokens.push({
          type: "text",
          content: "",
          range: Range.create(Position.create(lineIndex, lastIndex + offset + characterIndex), Position.create(lineIndex, lastIndex + offset + characterIndex))
        });
      }
      lastIndex = matchIndex + match[0].length;
    }
    const endContent = line.substring(lastIndex);
    if (endContent.length > 0) {
      tokens.push({
        type: "text",
        content: endContent,
        range: Range.create(Position.create(lineIndex, lastIndex + characterIndex), Position.create(lineIndex, lastIndex + characterIndex + endContent.length))
      });
    }
  }
  return tokens;
}
var nonWhitespaceRegex = /\S/;
var whitespaceEndRegex = /\s*$/;
function skipWhitespace(line, index) {
  const match = line.substring(index).match(nonWhitespaceRegex);
  if (match) {
    return index + match.index;
  } else {
    return line.length;
  }
}
function lastCharacter(line) {
  const match = line.match(whitespaceEndRegex);
  if (match && typeof match.index === "number") {
    return match.index;
  }
  return void 0;
}
function parseJSDocComment(context) {
  var _a, _b, _c, _d;
  const startPosition = Position.create(context.position.line, context.position.character);
  if (context.tokens.length === 0) {
    return new JSDocCommentImpl([], Range.create(startPosition, startPosition));
  }
  const elements = [];
  while (context.index < context.tokens.length) {
    const element = parseJSDocElement(context, elements[elements.length - 1]);
    if (element) {
      elements.push(element);
    }
  }
  const start = (_b = (_a = elements[0]) === null || _a === void 0 ? void 0 : _a.range.start) !== null && _b !== void 0 ? _b : startPosition;
  const end = (_d = (_c = elements[elements.length - 1]) === null || _c === void 0 ? void 0 : _c.range.end) !== null && _d !== void 0 ? _d : startPosition;
  return new JSDocCommentImpl(elements, Range.create(start, end));
}
function parseJSDocElement(context, last2) {
  const next = context.tokens[context.index];
  if (next.type === "tag") {
    return parseJSDocTag(context, false);
  } else if (next.type === "text" || next.type === "inline-tag") {
    return parseJSDocText(context);
  } else {
    appendEmptyLine(next, last2);
    context.index++;
    return void 0;
  }
}
function appendEmptyLine(token, element) {
  if (element) {
    const line = new JSDocLineImpl("", token.range);
    if ("inlines" in element) {
      element.inlines.push(line);
    } else {
      element.content.inlines.push(line);
    }
  }
}
function parseJSDocText(context) {
  let token = context.tokens[context.index];
  const firstToken = token;
  let lastToken = token;
  const lines = [];
  while (token && token.type !== "break" && token.type !== "tag") {
    lines.push(parseJSDocInline(context));
    lastToken = token;
    token = context.tokens[context.index];
  }
  return new JSDocTextImpl(lines, Range.create(firstToken.range.start, lastToken.range.end));
}
function parseJSDocInline(context) {
  const token = context.tokens[context.index];
  if (token.type === "inline-tag") {
    return parseJSDocTag(context, true);
  } else {
    return parseJSDocLine(context);
  }
}
function parseJSDocTag(context, inline) {
  const tagToken = context.tokens[context.index++];
  const name = tagToken.content.substring(1);
  const nextToken = context.tokens[context.index];
  if ((nextToken === null || nextToken === void 0 ? void 0 : nextToken.type) === "text") {
    if (inline) {
      const docLine = parseJSDocLine(context);
      return new JSDocTagImpl(name, new JSDocTextImpl([docLine], docLine.range), inline, Range.create(tagToken.range.start, docLine.range.end));
    } else {
      const textDoc = parseJSDocText(context);
      return new JSDocTagImpl(name, textDoc, inline, Range.create(tagToken.range.start, textDoc.range.end));
    }
  } else {
    const range2 = tagToken.range;
    return new JSDocTagImpl(name, new JSDocTextImpl([], range2), inline, range2);
  }
}
function parseJSDocLine(context) {
  const token = context.tokens[context.index++];
  return new JSDocLineImpl(token.content, token.range);
}
function normalizeOptions(options) {
  if (!options) {
    return normalizeOptions({
      start: "/**",
      end: "*/",
      line: "*"
    });
  }
  const { start, end, line } = options;
  return {
    start: normalizeOption(start, true),
    end: normalizeOption(end, false),
    line: normalizeOption(line, true)
  };
}
function normalizeOption(option2, start) {
  if (typeof option2 === "string" || typeof option2 === "object") {
    const escaped = typeof option2 === "string" ? escapeRegExp(option2) : option2.source;
    if (start) {
      return new RegExp(`^\\s*${escaped}`);
    } else {
      return new RegExp(`\\s*${escaped}\\s*$`);
    }
  } else {
    return option2;
  }
}
var JSDocCommentImpl = class {
  constructor(elements, range2) {
    this.elements = elements;
    this.range = range2;
  }
  getTag(name) {
    return this.getAllTags().find((e) => e.name === name);
  }
  getTags(name) {
    return this.getAllTags().filter((e) => e.name === name);
  }
  getAllTags() {
    return this.elements.filter((e) => "name" in e);
  }
  toString() {
    let value = "";
    for (const element of this.elements) {
      if (value.length === 0) {
        value = element.toString();
      } else {
        const text = element.toString();
        value += fillNewlines(value) + text;
      }
    }
    return value.trim();
  }
  toMarkdown(options) {
    let value = "";
    for (const element of this.elements) {
      if (value.length === 0) {
        value = element.toMarkdown(options);
      } else {
        const text = element.toMarkdown(options);
        value += fillNewlines(value) + text;
      }
    }
    return value.trim();
  }
};
var JSDocTagImpl = class {
  constructor(name, content, inline, range2) {
    this.name = name;
    this.content = content;
    this.inline = inline;
    this.range = range2;
  }
  toString() {
    let text = `@${this.name}`;
    const content = this.content.toString();
    if (this.content.inlines.length === 1) {
      text = `${text} ${content}`;
    } else if (this.content.inlines.length > 1) {
      text = `${text}
${content}`;
    }
    if (this.inline) {
      return `{${text}}`;
    } else {
      return text;
    }
  }
  toMarkdown(options) {
    var _a, _b;
    return (_b = (_a = options === null || options === void 0 ? void 0 : options.renderTag) === null || _a === void 0 ? void 0 : _a.call(options, this)) !== null && _b !== void 0 ? _b : this.toMarkdownDefault(options);
  }
  toMarkdownDefault(options) {
    const content = this.content.toMarkdown(options);
    if (this.inline) {
      const rendered = renderInlineTag(this.name, content, options !== null && options !== void 0 ? options : {});
      if (typeof rendered === "string") {
        return rendered;
      }
    }
    let marker = "";
    if ((options === null || options === void 0 ? void 0 : options.tag) === "italic" || (options === null || options === void 0 ? void 0 : options.tag) === void 0) {
      marker = "*";
    } else if ((options === null || options === void 0 ? void 0 : options.tag) === "bold") {
      marker = "**";
    } else if ((options === null || options === void 0 ? void 0 : options.tag) === "bold-italic") {
      marker = "***";
    }
    let text = `${marker}@${this.name}${marker}`;
    if (this.content.inlines.length === 1) {
      text = `${text} — ${content}`;
    } else if (this.content.inlines.length > 1) {
      text = `${text}
${content}`;
    }
    if (this.inline) {
      return `{${text}}`;
    } else {
      return text;
    }
  }
};
function renderInlineTag(tag, content, options) {
  var _a, _b;
  if (tag === "linkplain" || tag === "linkcode" || tag === "link") {
    const index = content.indexOf(" ");
    let display = content;
    if (index > 0) {
      const displayStart = skipWhitespace(content, index);
      display = content.substring(displayStart);
      content = content.substring(0, index);
    }
    if (tag === "linkcode" || tag === "link" && options.link === "code") {
      display = `\`${display}\``;
    }
    const renderedLink = (_b = (_a = options.renderLink) === null || _a === void 0 ? void 0 : _a.call(options, content, display)) !== null && _b !== void 0 ? _b : renderLinkDefault(content, display);
    return renderedLink;
  }
  return void 0;
}
function renderLinkDefault(content, display) {
  try {
    URI2.parse(content, true);
    return `[${display}](${content})`;
  } catch (_a) {
    return content;
  }
}
var JSDocTextImpl = class {
  constructor(lines, range2) {
    this.inlines = lines;
    this.range = range2;
  }
  toString() {
    let text = "";
    for (let i = 0; i < this.inlines.length; i++) {
      const inline = this.inlines[i];
      const next = this.inlines[i + 1];
      text += inline.toString();
      if (next && next.range.start.line > inline.range.start.line) {
        text += "\n";
      }
    }
    return text;
  }
  toMarkdown(options) {
    let text = "";
    for (let i = 0; i < this.inlines.length; i++) {
      const inline = this.inlines[i];
      const next = this.inlines[i + 1];
      text += inline.toMarkdown(options);
      if (next && next.range.start.line > inline.range.start.line) {
        text += "\n";
      }
    }
    return text;
  }
};
var JSDocLineImpl = class {
  constructor(text, range2) {
    this.text = text;
    this.range = range2;
  }
  toString() {
    return this.text;
  }
  toMarkdown() {
    return this.text;
  }
};
function fillNewlines(text) {
  if (text.endsWith("\n")) {
    return "\n";
  } else {
    return "\n\n";
  }
}

// node_modules/langium/lib/documentation/documentation-provider.js
var JSDocDocumentationProvider = class {
  constructor(services) {
    this.indexManager = services.shared.workspace.IndexManager;
    this.commentProvider = services.documentation.CommentProvider;
  }
  getDocumentation(node) {
    const comment = this.commentProvider.getComment(node);
    if (comment && isJSDoc(comment)) {
      const parsedJSDoc = parseJSDoc(comment);
      return parsedJSDoc.toMarkdown({
        renderLink: (link, display) => {
          return this.documentationLinkRenderer(node, link, display);
        },
        renderTag: (tag) => {
          return this.documentationTagRenderer(node, tag);
        }
      });
    }
    return void 0;
  }
  documentationLinkRenderer(node, name, display) {
    var _a;
    const description = (_a = this.findNameInPrecomputedScopes(node, name)) !== null && _a !== void 0 ? _a : this.findNameInGlobalScope(node, name);
    if (description && description.nameSegment) {
      const line = description.nameSegment.range.start.line + 1;
      const character = description.nameSegment.range.start.character + 1;
      const uri = description.documentUri.with({ fragment: `L${line},${character}` });
      return `[${display}](${uri.toString()})`;
    } else {
      return void 0;
    }
  }
  documentationTagRenderer(_node, _tag) {
    return void 0;
  }
  findNameInPrecomputedScopes(node, name) {
    const document = getDocument(node);
    const precomputed = document.precomputedScopes;
    if (!precomputed) {
      return void 0;
    }
    let currentNode = node;
    do {
      const allDescriptions = precomputed.get(currentNode);
      const description = allDescriptions.find((e) => e.name === name);
      if (description) {
        return description;
      }
      currentNode = currentNode.$container;
    } while (currentNode);
    return void 0;
  }
  findNameInGlobalScope(node, name) {
    const description = this.indexManager.allElements().find((e) => e.name === name);
    return description;
  }
};

// node_modules/langium/lib/documentation/comment-provider.js
var DefaultCommentProvider = class {
  constructor(services) {
    this.grammarConfig = () => services.parser.GrammarConfig;
  }
  getComment(node) {
    var _a;
    if (isAstNodeWithComment(node)) {
      return node.$comment;
    }
    return (_a = findCommentNode(node.$cstNode, this.grammarConfig().multilineCommentRules)) === null || _a === void 0 ? void 0 : _a.text;
  }
};

// node_modules/langium/lib/parser/async-parser.js
var DefaultAsyncParser = class {
  constructor(services) {
    this.syncParser = services.parser.LangiumParser;
  }
  parse(text, _cancelToken) {
    return Promise.resolve(this.syncParser.parse(text));
  }
};
var AbstractThreadedAsyncParser = class {
  constructor(services) {
    this.threadCount = 8;
    this.terminationDelay = 200;
    this.workerPool = [];
    this.queue = [];
    this.hydrator = services.serializer.Hydrator;
  }
  initializeWorkers() {
    while (this.workerPool.length < this.threadCount) {
      const worker = this.createWorker();
      worker.onReady(() => {
        if (this.queue.length > 0) {
          const deferred = this.queue.shift();
          if (deferred) {
            worker.lock();
            deferred.resolve(worker);
          }
        }
      });
      this.workerPool.push(worker);
    }
  }
  async parse(text, cancelToken) {
    const worker = await this.acquireParserWorker(cancelToken);
    const deferred = new Deferred();
    let timeout;
    const cancellation = cancelToken.onCancellationRequested(() => {
      timeout = setTimeout(() => {
        this.terminateWorker(worker);
      }, this.terminationDelay);
    });
    worker.parse(text).then((result2) => {
      const hydrated = this.hydrator.hydrate(result2);
      deferred.resolve(hydrated);
    }).catch((err) => {
      deferred.reject(err);
    }).finally(() => {
      cancellation.dispose();
      clearTimeout(timeout);
    });
    return deferred.promise;
  }
  terminateWorker(worker) {
    worker.terminate();
    const index = this.workerPool.indexOf(worker);
    if (index >= 0) {
      this.workerPool.splice(index, 1);
    }
  }
  async acquireParserWorker(cancelToken) {
    this.initializeWorkers();
    for (const worker of this.workerPool) {
      if (worker.ready) {
        worker.lock();
        return worker;
      }
    }
    const deferred = new Deferred();
    cancelToken.onCancellationRequested(() => {
      const index = this.queue.indexOf(deferred);
      if (index >= 0) {
        this.queue.splice(index, 1);
      }
      deferred.reject(OperationCancelled);
    });
    this.queue.push(deferred);
    return deferred.promise;
  }
};
var ParserWorker = class {
  get ready() {
    return this._ready;
  }
  get onReady() {
    return this.onReadyEmitter.event;
  }
  constructor(sendMessage, onMessage, onError, terminate) {
    this.onReadyEmitter = new event_exports.Emitter();
    this.deferred = new Deferred();
    this._ready = true;
    this._parsing = false;
    this.sendMessage = sendMessage;
    this._terminate = terminate;
    onMessage((result2) => {
      const parseResult = result2;
      this.deferred.resolve(parseResult);
      this.unlock();
    });
    onError((error) => {
      this.deferred.reject(error);
      this.unlock();
    });
  }
  terminate() {
    this.deferred.reject(OperationCancelled);
    this._terminate();
  }
  lock() {
    this._ready = false;
  }
  unlock() {
    this._parsing = false;
    this._ready = true;
    this.onReadyEmitter.fire();
  }
  parse(text) {
    if (this._parsing) {
      throw new Error("Parser worker is busy");
    }
    this._parsing = true;
    this.deferred = new Deferred();
    this.sendMessage(text);
    return this.deferred.promise;
  }
};

// node_modules/langium/lib/workspace/workspace-lock.js
var DefaultWorkspaceLock = class {
  constructor() {
    this.previousTokenSource = new cancellation_exports.CancellationTokenSource();
    this.writeQueue = [];
    this.readQueue = [];
    this.done = true;
  }
  write(action) {
    this.cancelWrite();
    const tokenSource = startCancelableOperation();
    this.previousTokenSource = tokenSource;
    return this.enqueue(this.writeQueue, action, tokenSource.token);
  }
  read(action) {
    return this.enqueue(this.readQueue, action);
  }
  enqueue(queue, action, cancellationToken = cancellation_exports.CancellationToken.None) {
    const deferred = new Deferred();
    const entry = {
      action,
      deferred,
      cancellationToken
    };
    queue.push(entry);
    this.performNextOperation();
    return deferred.promise;
  }
  async performNextOperation() {
    if (!this.done) {
      return;
    }
    const entries = [];
    if (this.writeQueue.length > 0) {
      entries.push(this.writeQueue.shift());
    } else if (this.readQueue.length > 0) {
      entries.push(...this.readQueue.splice(0, this.readQueue.length));
    } else {
      return;
    }
    this.done = false;
    await Promise.all(entries.map(async ({ action, deferred, cancellationToken }) => {
      try {
        const result2 = await Promise.resolve().then(() => action(cancellationToken));
        deferred.resolve(result2);
      } catch (err) {
        if (isOperationCancelled(err)) {
          deferred.resolve(void 0);
        } else {
          deferred.reject(err);
        }
      }
    }));
    this.done = true;
    this.performNextOperation();
  }
  cancelWrite() {
    this.previousTokenSource.cancel();
  }
};

// node_modules/langium/lib/serializer/hydrator.js
var DefaultHydrator = class {
  constructor(services) {
    this.grammarElementIdMap = new BiMap();
    this.tokenTypeIdMap = new BiMap();
    this.grammar = services.Grammar;
    this.lexer = services.parser.Lexer;
    this.linker = services.references.Linker;
  }
  dehydrate(result2) {
    return {
      lexerErrors: result2.lexerErrors,
      lexerReport: result2.lexerReport ? this.dehydrateLexerReport(result2.lexerReport) : void 0,
      // We need to create shallow copies of the errors
      // The original errors inherit from the `Error` class, which is not transferable across worker threads
      parserErrors: result2.parserErrors.map((e) => Object.assign(Object.assign({}, e), { message: e.message })),
      value: this.dehydrateAstNode(result2.value, this.createDehyrationContext(result2.value))
    };
  }
  dehydrateLexerReport(lexerReport) {
    return lexerReport;
  }
  createDehyrationContext(node) {
    const astNodes = /* @__PURE__ */ new Map();
    const cstNodes = /* @__PURE__ */ new Map();
    for (const astNode of streamAst(node)) {
      astNodes.set(astNode, {});
    }
    if (node.$cstNode) {
      for (const cstNode of streamCst(node.$cstNode)) {
        cstNodes.set(cstNode, {});
      }
    }
    return {
      astNodes,
      cstNodes
    };
  }
  dehydrateAstNode(node, context) {
    const obj = context.astNodes.get(node);
    obj.$type = node.$type;
    obj.$containerIndex = node.$containerIndex;
    obj.$containerProperty = node.$containerProperty;
    if (node.$cstNode !== void 0) {
      obj.$cstNode = this.dehydrateCstNode(node.$cstNode, context);
    }
    for (const [name, value] of Object.entries(node)) {
      if (name.startsWith("$")) {
        continue;
      }
      if (Array.isArray(value)) {
        const arr = [];
        obj[name] = arr;
        for (const item of value) {
          if (isAstNode(item)) {
            arr.push(this.dehydrateAstNode(item, context));
          } else if (isReference(item)) {
            arr.push(this.dehydrateReference(item, context));
          } else {
            arr.push(item);
          }
        }
      } else if (isAstNode(value)) {
        obj[name] = this.dehydrateAstNode(value, context);
      } else if (isReference(value)) {
        obj[name] = this.dehydrateReference(value, context);
      } else if (value !== void 0) {
        obj[name] = value;
      }
    }
    return obj;
  }
  dehydrateReference(reference, context) {
    const obj = {};
    obj.$refText = reference.$refText;
    if (reference.$refNode) {
      obj.$refNode = context.cstNodes.get(reference.$refNode);
    }
    return obj;
  }
  dehydrateCstNode(node, context) {
    const cstNode = context.cstNodes.get(node);
    if (isRootCstNode(node)) {
      cstNode.fullText = node.fullText;
    } else {
      cstNode.grammarSource = this.getGrammarElementId(node.grammarSource);
    }
    cstNode.hidden = node.hidden;
    cstNode.astNode = context.astNodes.get(node.astNode);
    if (isCompositeCstNode(node)) {
      cstNode.content = node.content.map((child) => this.dehydrateCstNode(child, context));
    } else if (isLeafCstNode(node)) {
      cstNode.tokenType = node.tokenType.name;
      cstNode.offset = node.offset;
      cstNode.length = node.length;
      cstNode.startLine = node.range.start.line;
      cstNode.startColumn = node.range.start.character;
      cstNode.endLine = node.range.end.line;
      cstNode.endColumn = node.range.end.character;
    }
    return cstNode;
  }
  hydrate(result2) {
    const node = result2.value;
    const context = this.createHydrationContext(node);
    if ("$cstNode" in node) {
      this.hydrateCstNode(node.$cstNode, context);
    }
    return {
      lexerErrors: result2.lexerErrors,
      lexerReport: result2.lexerReport,
      parserErrors: result2.parserErrors,
      value: this.hydrateAstNode(node, context)
    };
  }
  createHydrationContext(node) {
    const astNodes = /* @__PURE__ */ new Map();
    const cstNodes = /* @__PURE__ */ new Map();
    for (const astNode of streamAst(node)) {
      astNodes.set(astNode, {});
    }
    let root2;
    if (node.$cstNode) {
      for (const cstNode of streamCst(node.$cstNode)) {
        let cst;
        if ("fullText" in cstNode) {
          cst = new RootCstNodeImpl(cstNode.fullText);
          root2 = cst;
        } else if ("content" in cstNode) {
          cst = new CompositeCstNodeImpl();
        } else if ("tokenType" in cstNode) {
          cst = this.hydrateCstLeafNode(cstNode);
        }
        if (cst) {
          cstNodes.set(cstNode, cst);
          cst.root = root2;
        }
      }
    }
    return {
      astNodes,
      cstNodes
    };
  }
  hydrateAstNode(node, context) {
    const astNode = context.astNodes.get(node);
    astNode.$type = node.$type;
    astNode.$containerIndex = node.$containerIndex;
    astNode.$containerProperty = node.$containerProperty;
    if (node.$cstNode) {
      astNode.$cstNode = context.cstNodes.get(node.$cstNode);
    }
    for (const [name, value] of Object.entries(node)) {
      if (name.startsWith("$")) {
        continue;
      }
      if (Array.isArray(value)) {
        const arr = [];
        astNode[name] = arr;
        for (const item of value) {
          if (isAstNode(item)) {
            arr.push(this.setParent(this.hydrateAstNode(item, context), astNode));
          } else if (isReference(item)) {
            arr.push(this.hydrateReference(item, astNode, name, context));
          } else {
            arr.push(item);
          }
        }
      } else if (isAstNode(value)) {
        astNode[name] = this.setParent(this.hydrateAstNode(value, context), astNode);
      } else if (isReference(value)) {
        astNode[name] = this.hydrateReference(value, astNode, name, context);
      } else if (value !== void 0) {
        astNode[name] = value;
      }
    }
    return astNode;
  }
  setParent(node, parent2) {
    node.$container = parent2;
    return node;
  }
  hydrateReference(reference, node, name, context) {
    return this.linker.buildReference(node, name, context.cstNodes.get(reference.$refNode), reference.$refText);
  }
  hydrateCstNode(cstNode, context, num = 0) {
    const cstNodeObj = context.cstNodes.get(cstNode);
    if (typeof cstNode.grammarSource === "number") {
      cstNodeObj.grammarSource = this.getGrammarElement(cstNode.grammarSource);
    }
    cstNodeObj.astNode = context.astNodes.get(cstNode.astNode);
    if (isCompositeCstNode(cstNodeObj)) {
      for (const child of cstNode.content) {
        const hydrated = this.hydrateCstNode(child, context, num++);
        cstNodeObj.content.push(hydrated);
      }
    }
    return cstNodeObj;
  }
  hydrateCstLeafNode(cstNode) {
    const tokenType = this.getTokenType(cstNode.tokenType);
    const offset = cstNode.offset;
    const length = cstNode.length;
    const startLine = cstNode.startLine;
    const startColumn = cstNode.startColumn;
    const endLine = cstNode.endLine;
    const endColumn = cstNode.endColumn;
    const hidden = cstNode.hidden;
    const node = new LeafCstNodeImpl(offset, length, {
      start: {
        line: startLine,
        character: startColumn
      },
      end: {
        line: endLine,
        character: endColumn
      }
    }, tokenType, hidden);
    return node;
  }
  getTokenType(name) {
    return this.lexer.definition[name];
  }
  getGrammarElementId(node) {
    if (!node) {
      return void 0;
    }
    if (this.grammarElementIdMap.size === 0) {
      this.createGrammarElementIdMap();
    }
    return this.grammarElementIdMap.get(node);
  }
  getGrammarElement(id) {
    if (this.grammarElementIdMap.size === 0) {
      this.createGrammarElementIdMap();
    }
    const element = this.grammarElementIdMap.getKey(id);
    return element;
  }
  createGrammarElementIdMap() {
    let id = 0;
    for (const element of streamAst(this.grammar)) {
      if (isAbstractElement(element)) {
        this.grammarElementIdMap.set(element, id++);
      }
    }
  }
};

// node_modules/langium/lib/default-module.js
function createDefaultCoreModule(context) {
  return {
    documentation: {
      CommentProvider: (services) => new DefaultCommentProvider(services),
      DocumentationProvider: (services) => new JSDocDocumentationProvider(services)
    },
    parser: {
      AsyncParser: (services) => new DefaultAsyncParser(services),
      GrammarConfig: (services) => createGrammarConfig(services),
      LangiumParser: (services) => createLangiumParser(services),
      CompletionParser: (services) => createCompletionParser(services),
      ValueConverter: () => new DefaultValueConverter(),
      TokenBuilder: () => new DefaultTokenBuilder(),
      Lexer: (services) => new DefaultLexer(services),
      ParserErrorMessageProvider: () => new LangiumParserErrorMessageProvider(),
      LexerErrorMessageProvider: () => new DefaultLexerErrorMessageProvider()
    },
    workspace: {
      AstNodeLocator: () => new DefaultAstNodeLocator(),
      AstNodeDescriptionProvider: (services) => new DefaultAstNodeDescriptionProvider(services),
      ReferenceDescriptionProvider: (services) => new DefaultReferenceDescriptionProvider(services)
    },
    references: {
      Linker: (services) => new DefaultLinker(services),
      NameProvider: () => new DefaultNameProvider(),
      ScopeProvider: (services) => new DefaultScopeProvider(services),
      ScopeComputation: (services) => new DefaultScopeComputation(services),
      References: (services) => new DefaultReferences(services)
    },
    serializer: {
      Hydrator: (services) => new DefaultHydrator(services),
      JsonSerializer: (services) => new DefaultJsonSerializer(services)
    },
    validation: {
      DocumentValidator: (services) => new DefaultDocumentValidator(services),
      ValidationRegistry: (services) => new ValidationRegistry(services)
    },
    shared: () => context.shared
  };
}
function createDefaultSharedCoreModule(context) {
  return {
    ServiceRegistry: (services) => new DefaultServiceRegistry(services),
    workspace: {
      LangiumDocuments: (services) => new DefaultLangiumDocuments(services),
      LangiumDocumentFactory: (services) => new DefaultLangiumDocumentFactory(services),
      DocumentBuilder: (services) => new DefaultDocumentBuilder(services),
      IndexManager: (services) => new DefaultIndexManager(services),
      WorkspaceManager: (services) => new DefaultWorkspaceManager(services),
      FileSystemProvider: (services) => context.fileSystemProvider(services),
      WorkspaceLock: () => new DefaultWorkspaceLock(),
      ConfigurationProvider: (services) => new DefaultConfigurationProvider(services)
    }
  };
}

// node_modules/langium/lib/dependency-injection.js
var Module;
(function(Module2) {
  Module2.merge = (m1, m2) => _merge(_merge({}, m1), m2);
})(Module || (Module = {}));
function inject(module1, module2, module3, module4, module5, module6, module7, module8, module9) {
  const module10 = [module1, module2, module3, module4, module5, module6, module7, module8, module9].reduce(_merge, {});
  return _inject(module10);
}
var isProxy = Symbol("isProxy");
function eagerLoad(item) {
  if (item && item[isProxy]) {
    for (const value of Object.values(item)) {
      eagerLoad(value);
    }
  }
  return item;
}
function _inject(module2, injector) {
  const proxy = new Proxy({}, {
    deleteProperty: () => false,
    set: () => {
      throw new Error("Cannot set property on injected service container");
    },
    get: (obj, prop) => {
      if (prop === isProxy) {
        return true;
      } else {
        return _resolve(obj, prop, module2, injector || proxy);
      }
    },
    getOwnPropertyDescriptor: (obj, prop) => (_resolve(obj, prop, module2, injector || proxy), Object.getOwnPropertyDescriptor(obj, prop)),
    // used by for..in
    has: (_, prop) => prop in module2,
    // used by ..in..
    ownKeys: () => [...Object.getOwnPropertyNames(module2)]
    // used by for..in
  });
  return proxy;
}
var __requested__ = Symbol();
function _resolve(obj, prop, module2, injector) {
  if (prop in obj) {
    if (obj[prop] instanceof Error) {
      throw new Error("Construction failure. Please make sure that your dependencies are constructable.", { cause: obj[prop] });
    }
    if (obj[prop] === __requested__) {
      throw new Error('Cycle detected. Please make "' + String(prop) + '" lazy. Visit https://langium.org/docs/reference/configuration-services/#resolving-cyclic-dependencies');
    }
    return obj[prop];
  } else if (prop in module2) {
    const value = module2[prop];
    obj[prop] = __requested__;
    try {
      obj[prop] = typeof value === "function" ? value(injector) : _inject(value, injector);
    } catch (error) {
      obj[prop] = error instanceof Error ? error : void 0;
      throw error;
    }
    return obj[prop];
  } else {
    return void 0;
  }
}
function _merge(target, source) {
  if (source) {
    for (const [key, value2] of Object.entries(source)) {
      if (value2 !== void 0) {
        const value1 = target[key];
        if (value1 !== null && value2 !== null && typeof value1 === "object" && typeof value2 === "object") {
          target[key] = _merge(value1, value2);
        } else {
          target[key] = value2;
        }
      }
    }
  }
  return target;
}

// node_modules/langium/lib/parser/indentation-aware.js
var indentationBuilderDefaultOptions = {
  indentTokenName: "INDENT",
  dedentTokenName: "DEDENT",
  whitespaceTokenName: "WS",
  ignoreIndentationDelimiters: []
};
var LexingMode;
(function(LexingMode2) {
  LexingMode2["REGULAR"] = "indentation-sensitive";
  LexingMode2["IGNORE_INDENTATION"] = "ignore-indentation";
})(LexingMode || (LexingMode = {}));
var IndentationAwareTokenBuilder = class extends DefaultTokenBuilder {
  constructor(options = indentationBuilderDefaultOptions) {
    super();
    this.indentationStack = [0];
    this.whitespaceRegExp = /[ \t]+/y;
    this.options = Object.assign(Object.assign({}, indentationBuilderDefaultOptions), options);
    this.indentTokenType = createToken({
      name: this.options.indentTokenName,
      pattern: this.indentMatcher.bind(this),
      line_breaks: false
    });
    this.dedentTokenType = createToken({
      name: this.options.dedentTokenName,
      pattern: this.dedentMatcher.bind(this),
      line_breaks: false
    });
  }
  buildTokens(grammar, options) {
    const tokenTypes = super.buildTokens(grammar, options);
    if (!isTokenTypeArray(tokenTypes)) {
      throw new Error("Invalid tokens built by default builder");
    }
    const { indentTokenName, dedentTokenName, whitespaceTokenName, ignoreIndentationDelimiters } = this.options;
    let dedent;
    let indent;
    let ws;
    const otherTokens = [];
    for (const tokenType of tokenTypes) {
      for (const [begin, end] of ignoreIndentationDelimiters) {
        if (tokenType.name === begin) {
          tokenType.PUSH_MODE = LexingMode.IGNORE_INDENTATION;
        } else if (tokenType.name === end) {
          tokenType.POP_MODE = true;
        }
      }
      if (tokenType.name === dedentTokenName) {
        dedent = tokenType;
      } else if (tokenType.name === indentTokenName) {
        indent = tokenType;
      } else if (tokenType.name === whitespaceTokenName) {
        ws = tokenType;
      } else {
        otherTokens.push(tokenType);
      }
    }
    if (!dedent || !indent || !ws) {
      throw new Error("Some indentation/whitespace tokens not found!");
    }
    if (ignoreIndentationDelimiters.length > 0) {
      const multiModeLexerDef = {
        modes: {
          [LexingMode.REGULAR]: [dedent, indent, ...otherTokens, ws],
          [LexingMode.IGNORE_INDENTATION]: [...otherTokens, ws]
        },
        defaultMode: LexingMode.REGULAR
      };
      return multiModeLexerDef;
    } else {
      return [dedent, indent, ws, ...otherTokens];
    }
  }
  flushLexingReport(text) {
    const result2 = super.flushLexingReport(text);
    return Object.assign(Object.assign({}, result2), { remainingDedents: this.flushRemainingDedents(text) });
  }
  /**
   * Helper function to check if the current position is the start of a new line.
   *
   * @param text The full input string.
   * @param offset The current position at which to check
   * @returns Whether the current position is the start of a new line
   */
  isStartOfLine(text, offset) {
    return offset === 0 || "\r\n".includes(text[offset - 1]);
  }
  /**
   * A helper function used in matching both indents and dedents.
   *
   * @param text The full input string.
   * @param offset The current position at which to attempt a match
   * @param tokens Previously scanned tokens
   * @param groups Token Groups
   * @returns The current and previous indentation levels and the matched whitespace
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  matchWhitespace(text, offset, tokens, groups) {
    var _a;
    this.whitespaceRegExp.lastIndex = offset;
    const match = this.whitespaceRegExp.exec(text);
    return {
      currIndentLevel: (_a = match === null || match === void 0 ? void 0 : match[0].length) !== null && _a !== void 0 ? _a : 0,
      prevIndentLevel: this.indentationStack.at(-1),
      match
    };
  }
  /**
   * Helper function to create an instance of an indentation token.
   *
   * @param tokenType Indent or dedent token type
   * @param text Full input string, used to calculate the line number
   * @param image The original image of the token (tabs or spaces)
   * @param offset Current position in the input string
   * @returns The indentation token instance
   */
  createIndentationTokenInstance(tokenType, text, image, offset) {
    const lineNumber = this.getLineNumber(text, offset);
    return createTokenInstance(tokenType, image, offset, offset + image.length, lineNumber, lineNumber, 1, image.length);
  }
  /**
   * Helper function to get the line number at a given offset.
   *
   * @param text Full input string, used to calculate the line number
   * @param offset Current position in the input string
   * @returns The line number at the given offset
   */
  getLineNumber(text, offset) {
    return text.substring(0, offset).split(/\r\n|\r|\n/).length;
  }
  /**
   * A custom pattern for matching indents
   *
   * @param text The full input string.
   * @param offset The offset at which to attempt a match
   * @param tokens Previously scanned tokens
   * @param groups Token Groups
   */
  indentMatcher(text, offset, tokens, groups) {
    if (!this.isStartOfLine(text, offset)) {
      return null;
    }
    const { currIndentLevel, prevIndentLevel, match } = this.matchWhitespace(text, offset, tokens, groups);
    if (currIndentLevel <= prevIndentLevel) {
      return null;
    }
    this.indentationStack.push(currIndentLevel);
    return match;
  }
  /**
   * A custom pattern for matching dedents
   *
   * @param text The full input string.
   * @param offset The offset at which to attempt a match
   * @param tokens Previously scanned tokens
   * @param groups Token Groups
   */
  dedentMatcher(text, offset, tokens, groups) {
    var _a, _b, _c, _d;
    if (!this.isStartOfLine(text, offset)) {
      return null;
    }
    const { currIndentLevel, prevIndentLevel, match } = this.matchWhitespace(text, offset, tokens, groups);
    if (currIndentLevel >= prevIndentLevel) {
      return null;
    }
    const matchIndentIndex = this.indentationStack.lastIndexOf(currIndentLevel);
    if (matchIndentIndex === -1) {
      this.diagnostics.push({
        severity: "error",
        message: `Invalid dedent level ${currIndentLevel} at offset: ${offset}. Current indentation stack: ${this.indentationStack}`,
        offset,
        length: (_b = (_a = match === null || match === void 0 ? void 0 : match[0]) === null || _a === void 0 ? void 0 : _a.length) !== null && _b !== void 0 ? _b : 0,
        line: this.getLineNumber(text, offset),
        column: 1
      });
      return null;
    }
    const numberOfDedents = this.indentationStack.length - matchIndentIndex - 1;
    const newlinesBeforeDedent = (_d = (_c = text.substring(0, offset).match(/[\r\n]+$/)) === null || _c === void 0 ? void 0 : _c[0].length) !== null && _d !== void 0 ? _d : 1;
    for (let i = 0; i < numberOfDedents; i++) {
      const token = this.createIndentationTokenInstance(
        this.dedentTokenType,
        text,
        "",
        // Dedents are 0-width tokens
        offset - (newlinesBeforeDedent - 1)
      );
      tokens.push(token);
      this.indentationStack.pop();
    }
    return null;
  }
  buildTerminalToken(terminal) {
    const tokenType = super.buildTerminalToken(terminal);
    const { indentTokenName, dedentTokenName, whitespaceTokenName } = this.options;
    if (tokenType.name === indentTokenName) {
      return this.indentTokenType;
    } else if (tokenType.name === dedentTokenName) {
      return this.dedentTokenType;
    } else if (tokenType.name === whitespaceTokenName) {
      return createToken({
        name: whitespaceTokenName,
        pattern: this.whitespaceRegExp,
        group: Lexer.SKIPPED
      });
    }
    return tokenType;
  }
  /**
   * Resets the indentation stack between different runs of the lexer
   *
   * @param text Full text that was tokenized
   * @returns Remaining dedent tokens to match all previous indents at the end of the file
   */
  flushRemainingDedents(text) {
    const remainingDedents = [];
    while (this.indentationStack.length > 1) {
      remainingDedents.push(this.createIndentationTokenInstance(this.dedentTokenType, text, "", text.length));
      this.indentationStack.pop();
    }
    this.indentationStack = [0];
    return remainingDedents;
  }
};
var IndentationAwareLexer = class extends DefaultLexer {
  constructor(services) {
    super(services);
    if (services.parser.TokenBuilder instanceof IndentationAwareTokenBuilder) {
      this.indentationTokenBuilder = services.parser.TokenBuilder;
    } else {
      throw new Error("IndentationAwareLexer requires an accompanying IndentationAwareTokenBuilder");
    }
  }
  tokenize(text, options = DEFAULT_TOKENIZE_OPTIONS) {
    const result2 = super.tokenize(text);
    const report = result2.report;
    if ((options === null || options === void 0 ? void 0 : options.mode) === "full") {
      result2.tokens.push(...report.remainingDedents);
    }
    report.remainingDedents = [];
    const { indentTokenType, dedentTokenType } = this.indentationTokenBuilder;
    const indentTokenIdx = indentTokenType.tokenTypeIdx;
    const dedentTokenIdx = dedentTokenType.tokenTypeIdx;
    const cleanTokens = [];
    const length = result2.tokens.length - 1;
    for (let i = 0; i < length; i++) {
      const token = result2.tokens[i];
      const nextToken = result2.tokens[i + 1];
      if (token.tokenTypeIdx === indentTokenIdx && nextToken.tokenTypeIdx === dedentTokenIdx) {
        i++;
        continue;
      }
      cleanTokens.push(token);
    }
    if (length >= 0) {
      cleanTokens.push(result2.tokens[length]);
    }
    result2.tokens = cleanTokens;
    return result2;
  }
};

// node_modules/langium/lib/utils/index.js
var utils_exports = {};
__export(utils_exports, {
  AstUtils: () => ast_utils_exports,
  BiMap: () => BiMap,
  Cancellation: () => cancellation_exports,
  ContextCache: () => ContextCache,
  CstUtils: () => cst_utils_exports,
  DONE_RESULT: () => DONE_RESULT,
  Deferred: () => Deferred,
  Disposable: () => Disposable,
  DisposableCache: () => DisposableCache,
  DocumentCache: () => DocumentCache,
  EMPTY_STREAM: () => EMPTY_STREAM,
  ErrorWithLocation: () => ErrorWithLocation,
  GrammarUtils: () => grammar_utils_exports,
  MultiMap: () => MultiMap,
  OperationCancelled: () => OperationCancelled,
  Reduction: () => Reduction,
  RegExpUtils: () => regexp_utils_exports,
  SimpleCache: () => SimpleCache,
  StreamImpl: () => StreamImpl,
  TreeStreamImpl: () => TreeStreamImpl,
  URI: () => URI2,
  UriUtils: () => UriUtils,
  WorkspaceCache: () => WorkspaceCache,
  assertUnreachable: () => assertUnreachable,
  delayNextTick: () => delayNextTick,
  interruptAndCheck: () => interruptAndCheck,
  isOperationCancelled: () => isOperationCancelled,
  loadGrammarFromJson: () => loadGrammarFromJson,
  setInterruptionPeriod: () => setInterruptionPeriod,
  startCancelableOperation: () => startCancelableOperation,
  stream: () => stream
});
__reExport(utils_exports, event_exports);

// node_modules/langium/lib/workspace/file-system-provider.js
var EmptyFileSystemProvider = class {
  readFile() {
    throw new Error("No file system is available.");
  }
  async readDirectory() {
    return [];
  }
};
var EmptyFileSystem = {
  fileSystemProvider: () => new EmptyFileSystemProvider()
};

// node_modules/langium/lib/utils/grammar-loader.js
var minimalGrammarModule = {
  Grammar: () => void 0,
  LanguageMetaData: () => ({
    caseInsensitive: false,
    fileExtensions: [".langium"],
    languageId: "langium"
  })
};
var minimalSharedGrammarModule = {
  AstReflection: () => new LangiumGrammarAstReflection()
};
function createMinimalGrammarServices() {
  const shared = inject(createDefaultSharedCoreModule(EmptyFileSystem), minimalSharedGrammarModule);
  const grammar = inject(createDefaultCoreModule({ shared }), minimalGrammarModule);
  shared.ServiceRegistry.register(grammar);
  return grammar;
}
function loadGrammarFromJson(json) {
  var _a;
  const services = createMinimalGrammarServices();
  const astNode = services.serializer.JsonSerializer.deserialize(json);
  services.shared.workspace.LangiumDocumentFactory.fromModel(astNode, URI2.parse(`memory://${(_a = astNode.name) !== null && _a !== void 0 ? _a : "grammar"}.langium`));
  return astNode;
}

// node_modules/langium/lib/index.js
__reExport(lib_exports, utils_exports);
export {
  AbstractAstReflection,
  AbstractCstNode,
  AbstractLangiumParser,
  AbstractParserErrorMessageProvider,
  AbstractThreadedAsyncParser,
  ast_utils_exports as AstUtils,
  BiMap,
  cancellation_exports as Cancellation,
  CompositeCstNodeImpl,
  ContextCache,
  CstNodeBuilder,
  cst_utils_exports as CstUtils,
  DEFAULT_TOKENIZE_OPTIONS,
  DONE_RESULT,
  DatatypeSymbol,
  DefaultAstNodeDescriptionProvider,
  DefaultAstNodeLocator,
  DefaultAsyncParser,
  DefaultCommentProvider,
  DefaultConfigurationProvider,
  DefaultDocumentBuilder,
  DefaultDocumentValidator,
  DefaultHydrator,
  DefaultIndexManager,
  DefaultJsonSerializer,
  DefaultLangiumDocumentFactory,
  DefaultLangiumDocuments,
  DefaultLexer,
  DefaultLexerErrorMessageProvider,
  DefaultLinker,
  DefaultNameProvider,
  DefaultReferenceDescriptionProvider,
  DefaultReferences,
  DefaultScopeComputation,
  DefaultScopeProvider,
  DefaultServiceRegistry,
  DefaultTokenBuilder,
  DefaultValueConverter,
  DefaultWorkspaceLock,
  DefaultWorkspaceManager,
  Deferred,
  Disposable,
  DisposableCache,
  DocumentCache,
  DocumentState,
  DocumentValidator,
  EMPTY_SCOPE,
  EMPTY_STREAM,
  EmptyFileSystem,
  EmptyFileSystemProvider,
  ErrorWithLocation,
  ast_exports as GrammarAST,
  grammar_utils_exports as GrammarUtils,
  IndentationAwareLexer,
  IndentationAwareTokenBuilder,
  JSDocDocumentationProvider,
  LangiumCompletionParser,
  LangiumParser,
  LangiumParserErrorMessageProvider,
  LeafCstNodeImpl,
  LexingMode,
  MapScope,
  Module,
  MultiMap,
  OperationCancelled,
  ParserWorker,
  Reduction,
  regexp_utils_exports as RegExpUtils,
  RootCstNodeImpl,
  SimpleCache,
  StreamImpl,
  StreamScope,
  TextDocument2 as TextDocument,
  TreeStreamImpl,
  URI2 as URI,
  UriUtils,
  ValidationCategory,
  ValidationRegistry,
  ValueConverter,
  WorkspaceCache,
  assertUnreachable,
  createCompletionParser,
  createDefaultCoreModule,
  createDefaultSharedCoreModule,
  createGrammarConfig,
  createLangiumParser,
  createParser,
  delayNextTick,
  diagnosticData,
  eagerLoad,
  getDiagnosticRange,
  indentationBuilderDefaultOptions,
  inject,
  interruptAndCheck,
  isAstNode,
  isAstNodeDescription,
  isAstNodeWithComment,
  isCompositeCstNode,
  isIMultiModeLexerDefinition,
  isJSDoc,
  isLeafCstNode,
  isLinkingError,
  isNamed,
  isOperationCancelled,
  isReference,
  isRootCstNode,
  isTokenTypeArray,
  isTokenTypeDictionary,
  loadGrammarFromJson,
  parseJSDoc,
  prepareLangiumParser,
  setInterruptionPeriod,
  startCancelableOperation,
  stream,
  toDiagnosticData,
  toDiagnosticSeverity
};
/*! Bundled license information:

lodash-es/lodash.default.js:
  (**
   * @license
   * Lodash (Custom Build) <https://lodash.com/>
   * Build: `lodash modularize exports="es" -o ./`
   * Copyright OpenJS Foundation and other contributors <https://openjsf.org/>
   * Released under MIT license <https://lodash.com/license>
   * Based on Underscore.js 1.8.3 <http://underscorejs.org/LICENSE>
   * Copyright Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
   *)

lodash-es/lodash.js:
  (**
   * @license
   * Lodash (Custom Build) <https://lodash.com/>
   * Build: `lodash modularize exports="es" -o ./`
   * Copyright OpenJS Foundation and other contributors <https://openjsf.org/>
   * Released under MIT license <https://lodash.com/license>
   * Based on Underscore.js 1.8.3 <http://underscorejs.org/LICENSE>
   * Copyright Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
   *)
*/
//# sourceMappingURL=langium.js.map

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path
from tqdm import tqdm
import numpy as np
from tqdm.auto import trange

from art.exceptions import EstimatorError
from art.summary_writer import SummaryWriter, SummaryWriterDefault
from art.utils import get_feature_index
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config

import collections
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType

import numpy as np
import torch


from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
import cv2 #type:ignore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils_tool import bright_judge
import os
import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import yolov5

"""Abstract Base Classes (ABCs) according to PEP 3119."""


def abstractmethod(funcobj):
    """A decorator indicating abstract methods.

    Requires that the metaclass is ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated unless all of its abstract methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.  abstractmethod() may be used to declare
    abstract methods for properties and descriptors.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    """
    funcobj.__isabstractmethod__ = True
    return funcobj


class abstractclassmethod(classmethod):
    """A decorator indicating abstract classmethods.

    Deprecated, use 'classmethod' with 'abstractmethod' instead.
    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractstaticmethod(staticmethod):
    """A decorator indicating abstract staticmethods.

    Deprecated, use 'staticmethod' with 'abstractmethod' instead.
    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractproperty(property):
    """A decorator indicating abstract properties.

    Deprecated, use 'property' with 'abstractmethod' instead.
    """

    __isabstractmethod__ = True


try:
    from _abc import (get_cache_token, _abc_init, _abc_register,# type: ignore
                      _abc_instancecheck, _abc_subclasscheck, _get_dump,
                      _reset_registry, _reset_caches)
except ImportError:
    from _py_abc import ABCMeta, get_cache_token
    ABCMeta.__module__ = 'abc'
else:
    class ABCMeta(type):
        """Metaclass for defining Abstract Base Classes (ABCs).

        Use this metaclass to create an ABC.  An ABC can be subclassed
        directly, and then acts as a mix-in class.  You can also register
        unrelated concrete classes (even built-in classes) and unrelated
        ABCs as 'virtual subclasses' -- these and their descendants will
        be considered subclasses of the registering ABC by the built-in
        issubclass() function, but the registering ABC won't show up in
        their MRO (Method Resolution Order) nor will method
        implementations defined by the registering ABC be callable (not
        even via super()).
        """
        def __new__(mcls, name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, name, bases, namespace, **kwargs)
            _abc_init(cls)
            return cls

        def register(cls, subclass):
            """Register a virtual subclass of an ABC.

            Returns the subclass, to allow usage as a class decorator.
            """
            return _abc_register(cls, subclass)

        def __instancecheck__(cls, instance):
            """Override for isinstance(instance, cls)."""
            return _abc_instancecheck(cls, instance)

        def __subclasscheck__(cls, subclass):
            """Override for issubclass(subclass, cls)."""
            return _abc_subclasscheck(cls, subclass)

        def _dump_registry(cls, file=None):
            """Debug helper to print the ABC registry."""
            print(f"Class: {cls.__module__}.{cls.__qualname__}", file=file)
            print(f"Inv. counter: {get_cache_token()}", file=file)
            (_abc_registry, _abc_cache, _abc_negative_cache,
             _abc_negative_cache_version) = _get_dump(cls)
            print(f"_abc_registry: {_abc_registry!r}", file=file)
            print(f"_abc_cache: {_abc_cache!r}", file=file)
            print(f"_abc_negative_cache: {_abc_negative_cache!r}", file=file)
            print(f"_abc_negative_cache_version: {_abc_negative_cache_version!r}",
                  file=file)

        def _abc_registry_clear(cls):
            """Clear the registry (for debugging or testing)."""
            _reset_registry(cls)

        def _abc_caches_clear(cls):
            """Clear the caches (for debugging or testing)."""
            _reset_caches(cls)


class ABC(metaclass=ABCMeta):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()

    __all__ = [
    # Super-special typing primitives.
    'Any',
    'Callable',
    'ClassVar',
    'Final',
    'ForwardRef',
    'Generic',
    'Literal',
    'Optional',
    'Protocol',
    'Tuple',
    'Type',
    'TypeVar',
    'Union',

    # ABCs (from collections.abc).
    'AbstractSet',  # collections.abc.Set.
    'ByteString',
    'Container',
    'ContextManager',
    'Hashable',
    'ItemsView',
    'Iterable',
    'Iterator',
    'KeysView',
    'Mapping',
    'MappingView',
    'MutableMapping',
    'MutableSequence',
    'MutableSet',
    'Sequence',
    'Sized',
    'ValuesView',
    'Awaitable',
    'AsyncIterator',
    'AsyncIterable',
    'Coroutine',
    'Collection',
    'AsyncGenerator',
    'AsyncContextManager',

    # Structural checks, a.k.a. protocols.
    'Reversible',
    'SupportsAbs',
    'SupportsBytes',
    'SupportsComplex',
    'SupportsFloat',
    'SupportsIndex',
    'SupportsInt',
    'SupportsRound',

    # Concrete collection types.
    'ChainMap',
    'Counter',
    'Deque',
    'Dict',
    'DefaultDict',
    'List',
    'OrderedDict',
    'Set',
    'FrozenSet',
    'NamedTuple',  # Not really a type.
    'TypedDict',  # Not really a type.
    'Generator',

    # One-off things.
    'AnyStr',
    'cast',
    'final',
    'get_args',
    'get_origin',
    'get_type_hints',
    'NewType',
    'no_type_check',
    'no_type_check_decorator',
    'NoReturn',
    'overload',
    'runtime_checkable',
    'Text',
    'TYPE_CHECKING',
]

# The pseudo-submodules 're' and 'io' are part of the public
# namespace, but excluded from __all__ because they might stomp on
# legitimate imports of those modules.


def _type_check(arg, msg, is_argument=True):
    """Check that the argument is a type, and return it (internal helper).

    As a special case, accept None and return type(None) instead. Also wrap strings
    into ForwardRef instances. Consider several corner cases, for example plain
    special forms like Union are not valid, while Union[int, str] is OK, etc.
    The msg argument is a human-readable error message, e.g::

        "Union[arg, ...]: arg should be a type."

    We append the repr() of the actual value (truncated to 100 chars).
    """
    invalid_generic_forms = (Generic, Protocol)
    if is_argument:
        invalid_generic_forms = invalid_generic_forms + (ClassVar, Final)

    if arg is None:
        return type(None)
    if isinstance(arg, str):
        return ForwardRef(arg)
    if (isinstance(arg, _GenericAlias) and
            arg.__origin__ in invalid_generic_forms):
        raise TypeError(f"{arg} is not valid as type argument")
    if (isinstance(arg, _SpecialForm) and arg not in (Any, NoReturn) or
            arg in (Generic, Protocol)):
        raise TypeError(f"Plain {arg} is not valid as type argument")
    if isinstance(arg, (type, TypeVar, ForwardRef)):
        return arg
    if not callable(arg):
        raise TypeError(f"{msg} Got {arg!r:.100}.")
    return arg


def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).

    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    if isinstance(obj, type):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return('...')
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


def _collect_type_vars(types):
    """Collect all type variable contained in types in order of
    first appearance (lexicographic order). For example::

        _collect_type_vars((T, List[S, T])) == (T, S)
    """
    tvars = []
    for t in types:
        if isinstance(t, TypeVar) and t not in tvars:
            tvars.append(t)
        if isinstance(t, _GenericAlias) and not t._special:
            tvars.extend([t for t in t.__parameters__ if t not in tvars])
    return tuple(tvars)


def _subs_tvars(tp, tvars, subs):
    """Substitute type variables 'tvars' with substitutions 'subs'.
    These two must have the same length.
    """
    if not isinstance(tp, _GenericAlias):
        return tp
    new_args = list(tp.__args__)
    for a, arg in enumerate(tp.__args__):
        if isinstance(arg, TypeVar):
            for i, tvar in enumerate(tvars):
                if arg == tvar:
                    new_args[a] = subs[i]
        else:
            new_args[a] = _subs_tvars(arg, tvars, subs)
    if tp.__origin__ is Union:
        return Union[tuple(new_args)]
    return tp.copy_with(tuple(new_args))


def _check_generic(cls, parameters):
    """Check correct count for parameters of a generic cls (internal helper).
    This gives a nice error message in case of count mismatch.
    """
    if not cls.__parameters__:
        raise TypeError(f"{cls} is not a generic class")
    alen = len(parameters)
    elen = len(cls.__parameters__)
    if alen != elen:
        raise TypeError(f"Too {'many' if alen > elen else 'few'} parameters for {cls};"
                        f" actual {alen}, expected {elen}")


def _remove_dups_flatten(parameters):
    """An internal helper for Union creation and substitution: flatten Unions
    among parameters, then remove duplicates.
    """
    # Flatten out Union[Union[...], ...].
    params = []
    for p in parameters:
        if isinstance(p, _GenericAlias) and p.__origin__ is Union:
            params.extend(p.__args__)
        elif isinstance(p, tuple) and len(p) > 0 and p[0] is Union:
            params.extend(p[1:])
        else:
            params.append(p)
    # Weed out strict duplicates, preserving the first of each occurrence.
    all_params = set(params)
    if len(all_params) < len(params):
        new_params = []
        for t in params:
            if t in all_params:
                new_params.append(t)
                all_params.remove(t)
        params = new_params
        assert not all_params, all_params
    return tuple(params)


_cleanups = []


def _tp_cache(func):
    """Internal wrapper caching __getitem__ of generic types with a fallback to
    original function for non-hashable arguments.
    """
    cached = functools.lru_cache()(func)
    _cleanups.append(cached.cache_clear)

    @functools.wraps(func)
    def inner(*args, **kwds):
        try:
            return cached(*args, **kwds)
        except TypeError:
            pass  # All real errors (not unhashable args) are raised below.
        return func(*args, **kwds)
    return inner


def _eval_type(t, globalns, localns):
    """Evaluate all forward references in the given type t.
    For use of globalns and localns see the docstring for get_type_hints().
    """
    if isinstance(t, ForwardRef):
        return t._evaluate(globalns, localns)
    if isinstance(t, _GenericAlias):
        ev_args = tuple(_eval_type(a, globalns, localns) for a in t.__args__)
        if ev_args == t.__args__:
            return t
        res = t.copy_with(ev_args)
        res._special = t._special
        return res
    return t


class _Final:
    """Mixin to prohibit subclassing"""

    __slots__ = ('__weakref__',)

    def __init_subclass__(self, /, *args, **kwds):
        if '_root' not in kwds:
            raise TypeError("Cannot subclass special typing classes")

class _Immutable:
    """Mixin to indicate that object should not be copied."""

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _SpecialForm(_Final, _Immutable, _root=True):
    """Internal indicator of special typing constructs.
    See _doc instance attribute for specific docs.
    """

    __slots__ = ('_name', '_doc')

    def __new__(cls, *args, **kwds):
        """Constructor.

        This only exists to give a better error message in case
        someone tries to subclass a special typing object (not a good idea).
        """
        if (len(args) == 3 and
                isinstance(args[0], str) and
                isinstance(args[1], tuple)):
            # Close enough.
            raise TypeError(f"Cannot subclass {cls!r}")
        return super().__new__(cls)

    def __init__(self, name, doc):
        self._name = name
        self._doc = doc

    def __eq__(self, other):
        if not isinstance(other, _SpecialForm):
            return NotImplemented
        return self._name == other._name

    def __hash__(self):
        return hash((self._name,))

    def __repr__(self):
        return 'typing.' + self._name

    def __reduce__(self):
        return self._name

    def __call__(self, *args, **kwds):
        raise TypeError(f"Cannot instantiate {self!r}")

    def __instancecheck__(self, obj):
        raise TypeError(f"{self} cannot be used with isinstance()")

    def __subclasscheck__(self, cls):
        raise TypeError(f"{self} cannot be used with issubclass()")

    @_tp_cache
    def __getitem__(self, parameters):
        if self._name in ('ClassVar', 'Final'):
            item = _type_check(parameters, f'{self._name} accepts only single type.')
            return _GenericAlias(self, (item,))
        if self._name == 'Union':
            if parameters == ():
                raise TypeError("Cannot take a Union of no types.")
            if not isinstance(parameters, tuple):
                parameters = (parameters,)
            msg = "Union[arg, ...]: each arg must be a type."
            parameters = tuple(_type_check(p, msg) for p in parameters)
            parameters = _remove_dups_flatten(parameters)
            if len(parameters) == 1:
                return parameters[0]
            return _GenericAlias(self, parameters)
        if self._name == 'Optional':
            arg = _type_check(parameters, "Optional[t] requires a single type.")
            return Union[arg, type(None)]
        if self._name == 'Literal':
            # There is no '_type_check' call because arguments to Literal[...] are
            # values, not types.
            return _GenericAlias(self, parameters)
        raise TypeError(f"{self} is not subscriptable")


Any = _SpecialForm('Any', doc=
    """Special type indicating an unconstrained type.

    - Any is compatible with every type.
    - Any assumed to have all methods.
    - All values assumed to be instances of Any.

    Note that all the above statements are true from the point of view of
    static type checkers. At runtime, Any should not be used with instance
    or class checks.
    """)

NoReturn = _SpecialForm('NoReturn', doc=
    """Special type indicating functions that never return.
    Example::

      from typing import NoReturn

      def stop() -> NoReturn:
          raise Exception('no way')

    This type is invalid in other positions, e.g., ``List[NoReturn]``
    will fail in static type checkers.
    """)

ClassVar = _SpecialForm('ClassVar', doc=
    """Special type construct to mark class variables.

    An annotation wrapped in ClassVar indicates that a given
    attribute is intended to be used as a class variable and
    should not be set on instances of that class. Usage::

      class Starship:
          stats: ClassVar[Dict[str, int]] = {} # class variable
          damage: int = 10                     # instance variable

    ClassVar accepts only types and cannot be further subscribed.

    Note that ClassVar is not a class itself, and should not
    be used with isinstance() or issubclass().
    """)

Final = _SpecialForm('Final', doc=
    """Special typing construct to indicate final names to type checkers.

    A final name cannot be re-assigned or overridden in a subclass.
    For example:

      MAX_SIZE: Final = 9000
      MAX_SIZE += 1  # Error reported by type checker

      class Connection:
          TIMEOUT: Final[int] = 10

      class FastConnector(Connection):
          TIMEOUT = 1  # Error reported by type checker

    There is no runtime checking of these properties.
    """)

Union = _SpecialForm('Union', doc=
    """Union type; Union[X, Y] means either X or Y.

    To define a union, use e.g. Union[int, str].  Details:
    - The arguments must be types and there must be at least one.
    - None as an argument is a special case and is replaced by
      type(None).
    - Unions of unions are flattened, e.g.::

        Union[Union[int, str], float] == Union[int, str, float]

    - Unions of a single argument vanish, e.g.::

        Union[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        Union[int, str, int] == Union[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        Union[int, str] == Union[str, int]

    - You cannot subclass or instantiate a union.
    - You can use Optional[X] as a shorthand for Union[X, None].
    """)

Optional = _SpecialForm('Optional', doc=
    """Optional type.

    Optional[X] is equivalent to Union[X, None].
    """)

Literal = _SpecialForm('Literal', doc=
    """Special typing form to define literal types (a.k.a. value types).

    This form can be used to indicate to type checkers that the corresponding
    variable or function parameter has a value equivalent to the provided
    literal (or one of several literals):

      def validate_simple(data: Any) -> Literal[True]:  # always returns True
          ...

      MODE = Literal['r', 'rb', 'w', 'wb']
      def open_helper(file: str, mode: MODE) -> str:
          ...

      open_helper('/some/path', 'r')  # Passes type check
      open_helper('/other/path', 'typo')  # Error in type checker

   Literal[...] cannot be subclassed. At runtime, an arbitrary value
   is allowed as type argument to Literal[...], but type checkers may
   impose restrictions.
    """)


class ForwardRef(_Final, _root=True):
    """Internal wrapper to hold a forward reference."""

    __slots__ = ('__forward_arg__', '__forward_code__',
                 '__forward_evaluated__', '__forward_value__',
                 '__forward_is_argument__')

    def __init__(self, arg, is_argument=True):
        if not isinstance(arg, str):
            raise TypeError(f"Forward reference must be a string -- got {arg!r}")
        try:
            code = compile(arg, '<string>', 'eval')
        except SyntaxError:
            raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
        self.__forward_arg__ = arg
        self.__forward_code__ = code
        self.__forward_evaluated__ = False
        self.__forward_value__ = None
        self.__forward_is_argument__ = is_argument

    def _evaluate(self, globalns, localns):
        if not self.__forward_evaluated__ or localns is not globalns:
            if globalns is None and localns is None:
                globalns = localns = {}
            elif globalns is None:
                globalns = localns
            elif localns is None:
                localns = globalns
            self.__forward_value__ = _type_check(
                eval(self.__forward_code__, globalns, localns),
                "Forward references must evaluate to types.",
                is_argument=self.__forward_is_argument__)
            self.__forward_evaluated__ = True
        return self.__forward_value__

    def __eq__(self, other):
        if not isinstance(other, ForwardRef):
            return NotImplemented
        if self.__forward_evaluated__ and other.__forward_evaluated__:
            return (self.__forward_arg__ == other.__forward_arg__ and
                    self.__forward_value__ == other.__forward_value__)
        return self.__forward_arg__ == other.__forward_arg__

    def __hash__(self):
        return hash(self.__forward_arg__)

    def __repr__(self):
        return f'ForwardRef({self.__forward_arg__!r})'


class TypeVar(_Final, _Immutable, _root=True):
    """Type variable.

    Usage::

      T = TypeVar('T')  # Can be anything
      A = TypeVar('A', str, bytes)  # Must be str or bytes

    Type variables exist primarily for the benefit of static type
    checkers.  They serve as the parameters for generic types as well
    as for generic function definitions.  See class Generic for more
    information on generic types.  Generic functions work as follows:

      def repeat(x: T, n: int) -> List[T]:
          '''Return a list containing n references to x.'''
          return [x]*n

      def longest(x: A, y: A) -> A:
          '''Return the longest of two strings.'''
          return x if len(x) >= len(y) else y

    The latter example's signature is essentially the overloading
    of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
    that if the arguments are instances of some subclass of str,
    the return type is still plain str.

    At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

    Type variables defined with covariant=True or contravariant=True
    can be used to declare covariant or contravariant generic types.
    See PEP 484 for more details. By default generic types are invariant
    in all type variables.

    Type variables can be introspected. e.g.:

      T.__name__ == 'T'
      T.__constraints__ == ()
      T.__covariant__ == False
      T.__contravariant__ = False
      A.__constraints__ == (str, bytes)

    Note that only type variables defined in global scope can be pickled.
    """

    __slots__ = ('__name__', '__bound__', '__constraints__',
                 '__covariant__', '__contravariant__')

    def __init__(self, name, *constraints, bound=None,
                 covariant=False, contravariant=False):
        self.__name__ = name
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        if constraints and bound is not None:
            raise TypeError("Constraints cannot be combined with bound=...")
        if constraints and len(constraints) == 1:
            raise TypeError("A single constraint is not allowed")
        msg = "TypeVar(name, constraint, ...): constraints must be types."
        self.__constraints__ = tuple(_type_check(t, msg) for t in constraints)
        if bound:
            self.__bound__ = _type_check(bound, "Bound must be a type.")
        else:
            self.__bound__ = None
        try:
            def_mod = sys._getframe(1).f_globals.get('__name__', '__main__')  # for pickling
        except (AttributeError, ValueError):
            def_mod = None
        if def_mod != 'typing':
            self.__module__ = def_mod

    def __repr__(self):
        if self.__covariant__:
            prefix = '+'
        elif self.__contravariant__:
            prefix = '-'
        else:
            prefix = '~'
        return prefix + self.__name__

    def __reduce__(self):
        return self.__name__


# Special typing constructs Union, Optional, Generic, Callable and Tuple
# use three special attributes for internal bookkeeping of generic types:
# * __parameters__ is a tuple of unique free type parameters of a generic
#   type, for example, Dict[T, T].__parameters__ == (T,);
# * __origin__ keeps a reference to a type that was subscripted,
#   e.g., Union[T, int].__origin__ == Union, or the non-generic version of
#   the type.
# * __args__ is a tuple of all arguments used in subscripting,
#   e.g., Dict[T, int].__args__ == (T, int).


# Mapping from non-generic type names that have a generic alias in typing
# but with a different name.
_normalize_alias = {'list': 'List',
                    'tuple': 'Tuple',
                    'dict': 'Dict',
                    'set': 'Set',
                    'frozenset': 'FrozenSet',
                    'deque': 'Deque',
                    'defaultdict': 'DefaultDict',
                    'type': 'Type',
                    'Set': 'AbstractSet'}

def _is_dunder(attr):
    return attr.startswith('__') and attr.endswith('__')


class _GenericAlias(_Final, _root=True):
    """The central part of internal API.

    This represents a generic version of type 'origin' with type arguments 'params'.
    There are two kind of these aliases: user defined and special. The special ones
    are wrappers around builtin collections and ABCs in collections.abc. These must
    have 'name' always set. If 'inst' is False, then the alias can't be instantiated,
    this is used by e.g. typing.List and typing.Dict.
    """
    def __init__(self, origin, params, *, inst=True, special=False, name=None):
        self._inst = inst
        self._special = special
        if special and name is None:
            orig_name = origin.__name__
            name = _normalize_alias.get(orig_name, orig_name)
        self._name = name
        if not isinstance(params, tuple):
            params = (params,)
        self.__origin__ = origin
        self.__args__ = tuple(... if a is _TypingEllipsis else
                              () if a is _TypingEmpty else
                              a for a in params)
        self.__parameters__ = _collect_type_vars(params)
        self.__slots__ = None  # This is not documented.
        if not name:
            self.__module__ = origin.__module__

    @_tp_cache
    def __getitem__(self, params):
        if self.__origin__ in (Generic, Protocol):
            # Can't subscript Generic[...] or Protocol[...].
            raise TypeError(f"Cannot subscript already-subscripted {self}")
        if not isinstance(params, tuple):
            params = (params,)
        msg = "Parameters to generic types must be types."
        params = tuple(_type_check(p, msg) for p in params)
        _check_generic(self, params)
        return _subs_tvars(self, self.__parameters__, params)

    def copy_with(self, params):
        # We don't copy self._special.
        return _GenericAlias(self.__origin__, params, name=self._name, inst=self._inst)

    def __repr__(self):
        if (self._name != 'Callable' or
                len(self.__args__) == 2 and self.__args__[0] is Ellipsis):
            if self._name:
                name = 'typing.' + self._name
            else:
                name = _type_repr(self.__origin__)
            if not self._special:
                args = f'[{", ".join([_type_repr(a) for a in self.__args__])}]'
            else:
                args = ''
            return (f'{name}{args}')
        if self._special:
            return 'typing.Callable'
        return (f'typing.Callable'
                f'[[{", ".join([_type_repr(a) for a in self.__args__[:-1]])}], '
                f'{_type_repr(self.__args__[-1])}]')

    def __eq__(self, other):
        if not isinstance(other, _GenericAlias):
            return NotImplemented
        if self.__origin__ != other.__origin__:
            return False
        if self.__origin__ is Union and other.__origin__ is Union:
            return frozenset(self.__args__) == frozenset(other.__args__)
        return self.__args__ == other.__args__

    def __hash__(self):
        if self.__origin__ is Union:
            return hash((Union, frozenset(self.__args__)))
        return hash((self.__origin__, self.__args__))

    def __call__(self, *args, **kwargs):
        if not self._inst:
            raise TypeError(f"Type {self._name} cannot be instantiated; "
                            f"use {self._name.lower()}() instead")
        result = self.__origin__(*args, **kwargs)
        try:
            result.__orig_class__ = self
        except AttributeError:
            pass
        return result

    def __mro_entries__(self, bases):
        if self._name:  # generic version of an ABC or built-in class
            res = []
            if self.__origin__ not in bases:
                res.append(self.__origin__)
            i = bases.index(self)
            if not any(isinstance(b, _GenericAlias) or issubclass(b, Generic)
                       for b in bases[i+1:]):
                res.append(Generic)
            return tuple(res)
        if self.__origin__ is Generic:
            if Protocol in bases:
                return ()
            i = bases.index(self)
            for b in bases[i+1:]:
                if isinstance(b, _GenericAlias) and b is not self:
                    return ()
        return (self.__origin__,)

    def __getattr__(self, attr):
        # We are careful for copy and pickle.
        # Also for simplicity we just don't relay all dunder names
        if '__origin__' in self.__dict__ and not _is_dunder(attr):
            return getattr(self.__origin__, attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, val):
        if _is_dunder(attr) or attr in ('_name', '_inst', '_special'):
            super().__setattr__(attr, val)
        else:
            setattr(self.__origin__, attr, val)

    def __instancecheck__(self, obj):
        return self.__subclasscheck__(type(obj))

    def __subclasscheck__(self, cls):
        if self._special:
            if not isinstance(cls, _GenericAlias):
                return issubclass(cls, self.__origin__)
            if cls._special:
                return issubclass(cls.__origin__, self.__origin__)
        raise TypeError("Subscripted generics cannot be used with"
                        " class and instance checks")

    def __reduce__(self):
        if self._special:
            return self._name

        if self._name:
            origin = globals()[self._name]
        else:
            origin = self.__origin__
        if (origin is Callable and
            not (len(self.__args__) == 2 and self.__args__[0] is Ellipsis)):
            args = list(self.__args__[:-1]), self.__args__[-1]
        else:
            args = tuple(self.__args__)
            if len(args) == 1 and not isinstance(args[0], tuple):
                args, = args
        return operator.getitem, (origin, args)


class _VariadicGenericAlias(_GenericAlias, _root=True):
    """Same as _GenericAlias above but for variadic aliases. Currently,
    this is used only by special internal aliases: Tuple and Callable.
    """
    def __getitem__(self, params):
        if self._name != 'Callable' or not self._special:
            return self.__getitem_inner__(params)
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Callable must be used as "
                            "Callable[[arg, ...], result].")
        args, result = params
        if args is Ellipsis:
            params = (Ellipsis, result)
        else:
            if not isinstance(args, list):
                raise TypeError(f"Callable[args, result]: args must be a list."
                                f" Got {args}")
            params = (tuple(args), result)
        return self.__getitem_inner__(params)

    @_tp_cache
    def __getitem_inner__(self, params):
        if self.__origin__ is tuple and self._special:
            if params == ():
                return self.copy_with((_TypingEmpty,))
            if not isinstance(params, tuple):
                params = (params,)
            if len(params) == 2 and params[1] is ...:
                msg = "Tuple[t, ...]: t must be a type."
                p = _type_check(params[0], msg)
                return self.copy_with((p, _TypingEllipsis))
            msg = "Tuple[t0, t1, ...]: each t must be a type."
            params = tuple(_type_check(p, msg) for p in params)
            return self.copy_with(params)
        if self.__origin__ is collections.abc.Callable and self._special:
            args, result = params
            msg = "Callable[args, result]: result must be a type."
            result = _type_check(result, msg)
            if args is Ellipsis:
                return self.copy_with((_TypingEllipsis, result))
            msg = "Callable[[arg, ...], result]: each arg must be a type."
            args = tuple(_type_check(arg, msg) for arg in args)
            params = args + (result,)
            return self.copy_with(params)
        return super().__getitem__(params)


class Generic:
    """Abstract base class for generic types.

    A generic type is typically declared by inheriting from
    this class parameterized with one or more type variables.
    For example, a generic mapping type might be defined as::

      class Mapping(Generic[KT, VT]):
          def __getitem__(self, key: KT) -> VT:
              ...
          # Etc.

    This class can then be used as follows::

      def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
          try:
              return mapping[key]
          except KeyError:
              return default
    """
    __slots__ = ()
    _is_protocol = False

    def __new__(cls, *args, **kwds):
        if cls in (Generic, Protocol):
            raise TypeError(f"Type {cls.__name__} cannot be instantiated; "
                            "it can be used only as a base class")
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            obj = super().__new__(cls)
        else:
            obj = super().__new__(cls, *args, **kwds)
        return obj

    @_tp_cache
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)
        if not params and cls is not Tuple:
            raise TypeError(
                f"Parameter list to {cls.__qualname__}[...] cannot be empty")
        msg = "Parameters to generic types must be types."
        params = tuple(_type_check(p, msg) for p in params)
        if cls in (Generic, Protocol):
            # Generic and Protocol can only be subscripted with unique type variables.
            if not all(isinstance(p, TypeVar) for p in params):
                raise TypeError(
                    f"Parameters to {cls.__name__}[...] must all be type variables")
            if len(set(params)) != len(params):
                raise TypeError(
                    f"Parameters to {cls.__name__}[...] must all be unique")
        else:
            # Subscripting a regular Generic subclass.
            _check_generic(cls, params)
        return _GenericAlias(cls, params)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        tvars = []
        if '__orig_bases__' in cls.__dict__:
            error = Generic in cls.__orig_bases__
        else:
            error = Generic in cls.__bases__ and cls.__name__ != 'Protocol'
        if error:
            raise TypeError("Cannot inherit from plain Generic")
        if '__orig_bases__' in cls.__dict__:
            tvars = _collect_type_vars(cls.__orig_bases__)
            # Look for Generic[T1, ..., Tn].
            # If found, tvars must be a subset of it.
            # If not found, tvars is it.
            # Also check for and reject plain Generic,
            # and reject multiple Generic[...].
            gvars = None
            for base in cls.__orig_bases__:
                if (isinstance(base, _GenericAlias) and
                        base.__origin__ is Generic):
                    if gvars is not None:
                        raise TypeError(
                            "Cannot inherit from Generic[...] multiple types.")
                    gvars = base.__parameters__
            if gvars is not None:
                tvarset = set(tvars)
                gvarset = set(gvars)
                if not tvarset <= gvarset:
                    s_vars = ', '.join(str(t) for t in tvars if t not in gvarset)
                    s_args = ', '.join(str(g) for g in gvars)
                    raise TypeError(f"Some type variables ({s_vars}) are"
                                    f" not listed in Generic[{s_args}]")
                tvars = gvars
        cls.__parameters__ = tuple(tvars)


class _TypingEmpty:
    """Internal placeholder for () or []. Used by TupleMeta and CallableMeta
    to allow empty list/tuple in specific places, without allowing them
    to sneak in where prohibited.
    """


class _TypingEllipsis:
    """Internal placeholder for ... (ellipsis)."""


_TYPING_INTERNALS = ['__parameters__', '__orig_bases__',  '__orig_class__',
                     '_is_protocol', '_is_runtime_protocol']

_SPECIAL_NAMES = ['__abstractmethods__', '__annotations__', '__dict__', '__doc__',
                  '__init__', '__module__', '__new__', '__slots__',
                  '__subclasshook__', '__weakref__']

# These special attributes will be not collected as protocol members.
EXCLUDED_ATTRIBUTES = _TYPING_INTERNALS + _SPECIAL_NAMES + ['_MutableMapping__marker']


def _get_protocol_attrs(cls):
    """Collect protocol members from a protocol class objects.

    This includes names actually defined in the class dictionary, as well
    as names that appear in annotations. Special names (above) are skipped.
    """
    attrs = set()
    for base in cls.__mro__[:-1]:  # without object
        if base.__name__ in ('Protocol', 'Generic'):
            continue
        annotations = getattr(base, '__annotations__', {})
        for attr in list(base.__dict__.keys()) + list(annotations.keys()):
            if not attr.startswith('_abc_') and attr not in EXCLUDED_ATTRIBUTES:
                attrs.add(attr)
    return attrs


def _is_callable_members_only(cls):
    # PEP 544 prohibits using issubclass() with protocols that have non-method members.
    return all(callable(getattr(cls, attr, None)) for attr in _get_protocol_attrs(cls))


def _no_init(self, *args, **kwargs):
    if type(self)._is_protocol:
        raise TypeError('Protocols cannot be instantiated')


def _allow_reckless_class_cheks():
    """Allow instnance and class checks for special stdlib modules.

    The abc and functools modules indiscriminately call isinstance() and
    issubclass() on the whole MRO of a user class, which may contain protocols.
    """
    try:
        return sys._getframe(3).f_globals['__name__'] in ['abc', 'functools']
    except (AttributeError, ValueError):  # For platforms without _getframe().
        return True


_PROTO_WHITELIST = {
    'collections.abc': [
        'Callable', 'Awaitable', 'Iterable', 'Iterator', 'AsyncIterable',
        'Hashable', 'Sized', 'Container', 'Collection', 'Reversible',
    ],
    'contextlib': ['AbstractContextManager', 'AbstractAsyncContextManager'],
}


class _ProtocolMeta(ABCMeta):
    # This metaclass is really unfortunate and exists only because of
    # the lack of __instancehook__.
    def __instancecheck__(cls, instance):
        # We need this method for situations where attributes are
        # assigned in __init__.
        if ((not getattr(cls, '_is_protocol', False) or
                _is_callable_members_only(cls)) and
                issubclass(instance.__class__, cls)):
            return True
        if cls._is_protocol:
            if all(hasattr(instance, attr) and
                    # All *methods* can be blocked by setting them to None.
                    (not callable(getattr(cls, attr, None)) or
                     getattr(instance, attr) is not None)
                    for attr in _get_protocol_attrs(cls)):
                return True
        return super().__instancecheck__(instance)


class Protocol(Generic, metaclass=_ProtocolMeta):
    """Base class for protocol classes.

    Protocol classes are defined as::

        class Proto(Protocol):
            def meth(self) -> int:
                ...

    Such classes are primarily used with static type checkers that recognize
    structural subtyping (static duck-typing), for example::

        class C:
            def meth(self) -> int:
                return 0

        def func(x: Proto) -> int:
            return x.meth()

        func(C())  # Passes static type check

    See PEP 544 for details. Protocol classes decorated with
    @typing.runtime_checkable act as simple-minded runtime protocols that check
    only the presence of given attributes, ignoring their type signatures.
    Protocol classes can be generic, they are defined as::

        class GenProto(Protocol[T]):
            def meth(self) -> T:
                ...
    """
    __slots__ = ()
    _is_protocol = True
    _is_runtime_protocol = False

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        # Determine if this is a protocol or a concrete subclass.
        if not cls.__dict__.get('_is_protocol', False):
            cls._is_protocol = any(b is Protocol for b in cls.__bases__)

        # Set (or override) the protocol subclass hook.
        def _proto_hook(other):
            if not cls.__dict__.get('_is_protocol', False):
                return NotImplemented

            # First, perform various sanity checks.
            if not getattr(cls, '_is_runtime_protocol', False):
                if _allow_reckless_class_cheks():
                    return NotImplemented
                raise TypeError("Instance and class checks can only be used with"
                                " @runtime_checkable protocols")
            if not _is_callable_members_only(cls):
                if _allow_reckless_class_cheks():
                    return NotImplemented
                raise TypeError("Protocols with non-method members"
                                " don't support issubclass()")
            if not isinstance(other, type):
                # Same error message as for issubclass(1, int).
                raise TypeError('issubclass() arg 1 must be a class')

            # Second, perform the actual structural compatibility check.
            for attr in _get_protocol_attrs(cls):
                for base in other.__mro__:
                    # Check if the members appears in the class dictionary...
                    if attr in base.__dict__:
                        if base.__dict__[attr] is None:
                            return NotImplemented
                        break

                    # ...or in annotations, if it is a sub-protocol.
                    annotations = getattr(base, '__annotations__', {})
                    if (isinstance(annotations, collections.abc.Mapping) and
                            attr in annotations and
                            issubclass(other, Generic) and other._is_protocol):
                        break
                else:
                    return NotImplemented
            return True

        if '__subclasshook__' not in cls.__dict__:
            cls.__subclasshook__ = _proto_hook

        # We have nothing more to do for non-protocols...
        if not cls._is_protocol:
            return

        # ... otherwise check consistency of bases, and prohibit instantiation.
        for base in cls.__bases__:
            if not (base in (object, Generic) or
                    base.__module__ in _PROTO_WHITELIST and
                    base.__name__ in _PROTO_WHITELIST[base.__module__] or
                    issubclass(base, Generic) and base._is_protocol):
                raise TypeError('Protocols can only inherit from other'
                                ' protocols, got %r' % base)
        cls.__init__ = _no_init


def runtime_checkable(cls):
    """Mark a protocol class as a runtime protocol.

    Such protocol can be used with isinstance() and issubclass().
    Raise TypeError if applied to a non-protocol class.
    This allows a simple-minded structural check very similar to
    one trick ponies in collections.abc such as Iterable.
    For example::

        @runtime_checkable
        class Closable(Protocol):
            def close(self): ...

        assert isinstance(open('/some/file'), Closable)

    Warning: this will check only the presence of the required methods,
    not their type signatures!
    """
    if not issubclass(cls, Generic) or not cls._is_protocol:
        raise TypeError('@runtime_checkable can be only applied to protocol classes,'
                        ' got %r' % cls)
    cls._is_runtime_protocol = True
    return cls


def cast(typ, val):
    """Cast a value to a type.

    This returns the value unchanged.  To the type checker this
    signals that the return value has the designated type, but at
    runtime we intentionally don't check anything (we want this
    to be as fast as possible).
    """
    return val


def _get_defaults(func):
    """Internal helper to extract the default arguments, by name."""
    try:
        code = func.__code__
    except AttributeError:
        # Some built-in functions don't have __code__, __defaults__, etc.
        return {}
    pos_count = code.co_argcount
    arg_names = code.co_varnames
    arg_names = arg_names[:pos_count]
    defaults = func.__defaults__ or ()
    kwdefaults = func.__kwdefaults__
    res = dict(kwdefaults) if kwdefaults else {}
    pos_offset = pos_count - len(defaults)
    for name, value in zip(arg_names[pos_offset:], defaults):
        assert name not in res
        res[name] = value
    return res


_allowed_types = (types.FunctionType, types.BuiltinFunctionType,
                  types.MethodType, types.ModuleType,
                  WrapperDescriptorType, MethodWrapperType, MethodDescriptorType)


def get_type_hints(obj, globalns=None, localns=None):
    """Return type hints for an object.

    This is often the same as obj.__annotations__, but it handles
    forward references encoded as string literals, and if necessary
    adds Optional[t] if a default value equal to None is set.

    The argument may be a module, class, method, or function. The annotations
    are returned as a dictionary. For classes, annotations include also
    inherited members.

    TypeError is raised if the argument is not of a type that can contain
    annotations, and an empty dictionary is returned if no annotations are
    present.

    BEWARE -- the behavior of globalns and localns is counterintuitive
    (unless you are familiar with how eval() and exec() work).  The
    search order is locals first, then globals.

    - If no dict arguments are passed, an attempt is made to use the
      globals from obj (or the respective module's globals for classes),
      and these are also used as the locals.  If the object does not appear
      to have globals, an empty dictionary is used.

    - If one dict argument is passed, it is used for both globals and
      locals.

    - If two dict arguments are passed, they specify globals and
      locals, respectively.
    """

    if getattr(obj, '__no_type_check__', None):
        return {}
    # Classes require a special treatment.
    if isinstance(obj, type):
        hints = {}
        for base in reversed(obj.__mro__):
            if globalns is None:
                base_globals = sys.modules[base.__module__].__dict__
            else:
                base_globals = globalns
            ann = base.__dict__.get('__annotations__', {})
            for name, value in ann.items():
                if value is None:
                    value = type(None)
                if isinstance(value, str):
                    value = ForwardRef(value, is_argument=False)
                value = _eval_type(value, base_globals, localns)
                hints[name] = value
        return hints

    if globalns is None:
        if isinstance(obj, types.ModuleType):
            globalns = obj.__dict__
        else:
            nsobj = obj
            # Find globalns for the unwrapped object.
            while hasattr(nsobj, '__wrapped__'):
                nsobj = nsobj.__wrapped__
            globalns = getattr(nsobj, '__globals__', {})
        if localns is None:
            localns = globalns
    elif localns is None:
        localns = globalns
    hints = getattr(obj, '__annotations__', None)
    if hints is None:
        # Return empty annotations for something that _could_ have them.
        if isinstance(obj, _allowed_types):
            return {}
        else:
            raise TypeError('{!r} is not a module, class, method, '
                            'or function.'.format(obj))
    defaults = _get_defaults(obj)
    hints = dict(hints)
    for name, value in hints.items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            value = ForwardRef(value)
        value = _eval_type(value, globalns, localns)
        if name in defaults and defaults[name] is None:
            value = Optional[value]
        hints[name] = value
    return hints


def get_origin(tp):
    """Get the unsubscripted version of a type.

    This supports generic types, Callable, Tuple, Union, Literal, Final and ClassVar.
    Return None for unsupported types. Examples::

        get_origin(Literal[42]) is Literal
        get_origin(int) is None
        get_origin(ClassVar[int]) is ClassVar
        get_origin(Generic) is Generic
        get_origin(Generic[T]) is Generic
        get_origin(Union[T, int]) is Union
        get_origin(List[Tuple[T, T]][int]) == list
    """
    if isinstance(tp, _GenericAlias):
        return tp.__origin__
    if tp is Generic:
        return Generic
    return None


def get_args(tp):
    """Get type arguments with all substitutions performed.

    For unions, basic simplifications used by Union constructor are performed.
    Examples::
        get_args(Dict[str, int]) == (str, int)
        get_args(int) == ()
        get_args(Union[int, Union[T, int], str][int]) == (int, str)
        get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
        get_args(Callable[[], T][int]) == ([], int)
    """
    if isinstance(tp, _GenericAlias) and not tp._special:
        res = tp.__args__
        if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
            res = (list(res[:-1]), res[-1])
        return res
    return ()


def no_type_check(arg):
    """Decorator to indicate that annotations are not type hints.

    The argument must be a class or function; if it is a class, it
    applies recursively to all methods and classes defined in that class
    (but not to methods defined in its superclasses or subclasses).

    This mutates the function(s) or class(es) in place.
    """
    if isinstance(arg, type):
        arg_attrs = arg.__dict__.copy()
        for attr, val in arg.__dict__.items():
            if val in arg.__bases__ + (arg,):
                arg_attrs.pop(attr)
        for obj in arg_attrs.values():
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True
            if isinstance(obj, type):
                no_type_check(obj)
    try:
        arg.__no_type_check__ = True
    except TypeError:  # built-in classes
        pass
    return arg


def no_type_check_decorator(decorator):
    """Decorator to give another decorator the @no_type_check effect.

    This wraps the decorator with something that wraps the decorated
    function in @no_type_check.
    """

    @functools.wraps(decorator)
    def wrapped_decorator(*args, **kwds):
        func = decorator(*args, **kwds)
        func = no_type_check(func)
        return func

    return wrapped_decorator


def _overload_dummy(*args, **kwds):
    """Helper for @overload to raise when called."""
    raise NotImplementedError(
        "You should not call an overloaded function. "
        "A series of @overload-decorated functions "
        "outside a stub module should always be followed "
        "by an implementation that is not @overload-ed.")


def overload(func):
    """Decorator for overloaded functions/methods.

    In a stub file, place two or more stub definitions for the same
    function in a row, each decorated with @overload.  For example:

      @overload
      def utf8(value: None) -> None: ...
      @overload
      def utf8(value: bytes) -> bytes: ...
      @overload
      def utf8(value: str) -> bytes: ...

    In a non-stub file (i.e. a regular .py file), do the same but
    follow it with an implementation.  The implementation should *not*
    be decorated with @overload.  For example:

      @overload
      def utf8(value: None) -> None: ...
      @overload
      def utf8(value: bytes) -> bytes: ...
      @overload
      def utf8(value: str) -> bytes: ...
      def utf8(value):
          # implementation goes here
    """
    return _overload_dummy


def final(f):
    """A decorator to indicate final methods and final classes.

    Use this decorator to indicate to type checkers that the decorated
    method cannot be overridden, and decorated class cannot be subclassed.
    For example:

      class Base:
          @final
          def done(self) -> None:
              ...
      class Sub(Base):
          def done(self) -> None:  # Error reported by type checker
                ...

      @final
      class Leaf:
          ...
      class Other(Leaf):  # Error reported by type checker
          ...

    There is no runtime checking of these properties.
    """
    return f


# Some unconstrained type variables.  These are used by the container types.
# (These are not for export.)
T = TypeVar('T')  # Any type.
KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
T_co = TypeVar('T_co', covariant=True)  # Any type covariant containers.
V_co = TypeVar('V_co', covariant=True)  # Any type covariant containers.
VT_co = TypeVar('VT_co', covariant=True)  # Value type covariant containers.
T_contra = TypeVar('T_contra', contravariant=True)  # Ditto contravariant.
# Internal type variable used for Type[].
CT_co = TypeVar('CT_co', covariant=True, bound=type)

# A useful type variable with constraints.  This represents string types.
# (This one *is* for export!)
AnyStr = TypeVar('AnyStr', bytes, str)


# Various ABCs mimicking those in collections.abc.
def _alias(origin, params, inst=True):
    return _GenericAlias(origin, params, special=True, inst=inst)

Hashable = _alias(collections.abc.Hashable, ())  # Not generic.
Awaitable = _alias(collections.abc.Awaitable, T_co)
Coroutine = _alias(collections.abc.Coroutine, (T_co, T_contra, V_co))
AsyncIterable = _alias(collections.abc.AsyncIterable, T_co)
AsyncIterator = _alias(collections.abc.AsyncIterator, T_co)
Iterable = _alias(collections.abc.Iterable, T_co)
Iterator = _alias(collections.abc.Iterator, T_co)
Reversible = _alias(collections.abc.Reversible, T_co)
Sized = _alias(collections.abc.Sized, ())  # Not generic.
Container = _alias(collections.abc.Container, T_co)
Collection = _alias(collections.abc.Collection, T_co)
Callable = _VariadicGenericAlias(collections.abc.Callable, (), special=True)
Callable.__doc__ = \
    """Callable type; Callable[[int], str] is a function of (int) -> str.

    The subscription syntax must always be used with exactly two
    values: the argument list and the return type.  The argument list
    must be a list of types or ellipsis; the return type must be a single type.

    There is no syntax to indicate optional or keyword arguments,
    such function types are rarely used as callback types.
    """
AbstractSet = _alias(collections.abc.Set, T_co)
MutableSet = _alias(collections.abc.MutableSet, T)
# NOTE: Mapping is only covariant in the value type.
Mapping = _alias(collections.abc.Mapping, (KT, VT_co))
MutableMapping = _alias(collections.abc.MutableMapping, (KT, VT))
Sequence = _alias(collections.abc.Sequence, T_co)
MutableSequence = _alias(collections.abc.MutableSequence, T)
ByteString = _alias(collections.abc.ByteString, ())  # Not generic
Tuple = _VariadicGenericAlias(tuple, (), inst=False, special=True)
Tuple.__doc__ = \
    """Tuple type; Tuple[X, Y] is the cross-product type of X and Y.

    Example: Tuple[T1, T2] is a tuple of two elements corresponding
    to type variables T1 and T2.  Tuple[int, float, str] is a tuple
    of an int, a float and a string.

    To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].
    """
List = _alias(list, T, inst=False)
Deque = _alias(collections.deque, T)
Set = _alias(set, T, inst=False)
FrozenSet = _alias(frozenset, T_co, inst=False)
MappingView = _alias(collections.abc.MappingView, T_co)
KeysView = _alias(collections.abc.KeysView, KT)
ItemsView = _alias(collections.abc.ItemsView, (KT, VT_co))
ValuesView = _alias(collections.abc.ValuesView, VT_co)
ContextManager = _alias(contextlib.AbstractContextManager, T_co)
AsyncContextManager = _alias(contextlib.AbstractAsyncContextManager, T_co)
Dict = _alias(dict, (KT, VT), inst=False)
DefaultDict = _alias(collections.defaultdict, (KT, VT))
OrderedDict = _alias(collections.OrderedDict, (KT, VT))
Counter = _alias(collections.Counter, T)
ChainMap = _alias(collections.ChainMap, (KT, VT))
Generator = _alias(collections.abc.Generator, (T_co, T_contra, V_co))
AsyncGenerator = _alias(collections.abc.AsyncGenerator, (T_co, T_contra))
Type = _alias(type, CT_co, inst=False)
Type.__doc__ = \
    """A special construct usable to annotate class objects.

    For example, suppose we have the following classes::

      class User: ...  # Abstract base for User classes
      class BasicUser(User): ...
      class ProUser(User): ...
      class TeamUser(User): ...

    And a function that takes a class argument that's a subclass of
    User and returns an instance of the corresponding class::

      U = TypeVar('U', bound=User)
      def new_user(user_class: Type[U]) -> U:
          user = user_class()
          # (Here we could write the user object to a database)
          return user

      joe = new_user(BasicUser)

    At this point the type checker knows that joe has type BasicUser.
    """


@runtime_checkable
class SupportsInt(Protocol):
    """An ABC with one abstract method __int__."""
    __slots__ = ()

    @abstractmethod
    def __int__(self) -> int:
        pass


@runtime_checkable
class SupportsFloat(Protocol):
    """An ABC with one abstract method __float__."""
    __slots__ = ()

    @abstractmethod
    def __float__(self) -> float:
        pass


@runtime_checkable
class SupportsComplex(Protocol):
    """An ABC with one abstract method __complex__."""
    __slots__ = ()

    @abstractmethod
    def __complex__(self) -> complex:
        pass


@runtime_checkable
class SupportsBytes(Protocol):
    """An ABC with one abstract method __bytes__."""
    __slots__ = ()

    @abstractmethod
    def __bytes__(self) -> bytes:
        pass


@runtime_checkable
class SupportsIndex(Protocol):
    """An ABC with one abstract method __index__."""
    __slots__ = ()

    @abstractmethod
    def __index__(self) -> int:
        pass


@runtime_checkable
class SupportsAbs(Protocol[T_co]):
    """An ABC with one abstract method __abs__ that is covariant in its return type."""
    __slots__ = ()

    @abstractmethod
    def __abs__(self) -> T_co:# type: ignore
        pass


@runtime_checkable
class SupportsRound(Protocol[T_co]):
    """An ABC with one abstract method __round__ that is covariant in its return type."""
    __slots__ = ()

    @abstractmethod
    def __round__(self, ndigits: int = 0) -> T_co:# type: ignore
        pass


def _make_nmtuple(name, types):
    msg = "NamedTuple('Name', [(f0, t0), (f1, t1), ...]); each t must be a type"
    types = [(n, _type_check(t, msg)) for n, t in types]
    nm_tpl = collections.namedtuple(name, [n for n, t in types])
    # Prior to PEP 526, only _field_types attribute was assigned.
    # Now __annotations__ are used and _field_types is deprecated (remove in 3.9)
    nm_tpl.__annotations__ = nm_tpl._field_types = dict(types)
    try:
        nm_tpl.__module__ = sys._getframe(2).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return nm_tpl


# attributes prohibited to set in NamedTuple class syntax
_prohibited = ('__new__', '__init__', '__slots__', '__getnewargs__',
               '_fields', '_field_defaults', '_field_types',
               '_make', '_replace', '_asdict', '_source')

_special = ('__module__', '__name__', '__annotations__')


class NamedTupleMeta(type):

    def __new__(cls, typename, bases, ns):
        if ns.get('_root', False):
            return super().__new__(cls, typename, bases, ns)
        types = ns.get('__annotations__', {})
        nm_tpl = _make_nmtuple(typename, types.items())
        defaults = []
        defaults_dict = {}
        for field_name in types:
            if field_name in ns:
                default_value = ns[field_name]
                defaults.append(default_value)
                defaults_dict[field_name] = default_value
            elif defaults:
                raise TypeError("Non-default namedtuple field {field_name} cannot "
                                "follow default field(s) {default_names}"
                                .format(field_name=field_name,
                                        default_names=', '.join(defaults_dict.keys())))
        nm_tpl.__new__.__annotations__ = dict(types)
        nm_tpl.__new__.__defaults__ = tuple(defaults)
        nm_tpl._field_defaults = defaults_dict
        # update from user namespace without overriding special namedtuple attributes
        for key in ns:
            if key in _prohibited:
                raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
            elif key not in _special and key not in nm_tpl._fields:
                setattr(nm_tpl, key, ns[key])
        return nm_tpl


class NamedTuple(metaclass=NamedTupleMeta):
    """Typed version of namedtuple.

    Usage in Python versions >= 3.6::

        class Employee(NamedTuple):
            name: str
            id: int

    This is equivalent to::

        Employee = collections.namedtuple('Employee', ['name', 'id'])

    The resulting class has an extra __annotations__ attribute, giving a
    dict that maps field names to types.  (The field names are also in
    the _fields attribute, which is part of the namedtuple API.)
    Alternative equivalent keyword syntax is also accepted::

        Employee = NamedTuple('Employee', name=str, id=int)

    In Python versions <= 3.5 use::

        Employee = NamedTuple('Employee', [('name', str), ('id', int)])
    """
    _root = True

    def __new__(*args, **kwargs):
        if not args:
            raise TypeError('NamedTuple.__new__(): not enough arguments')
        cls, *args = args  # allow the "cls" keyword be passed
        if args:
            typename, *args = args # allow the "typename" keyword be passed
        elif 'typename' in kwargs:
            typename = kwargs.pop('typename')
            import warnings
            warnings.warn("Passing 'typename' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            raise TypeError("NamedTuple.__new__() missing 1 required positional "
                            "argument: 'typename'")
        if args:
            try:
                fields, = args # allow the "fields" keyword be passed
            except ValueError:
                raise TypeError(f'NamedTuple.__new__() takes from 2 to 3 '
                                f'positional arguments but {len(args) + 2} '
                                f'were given') from None
        elif 'fields' in kwargs and len(kwargs) == 1:
            fields = kwargs.pop('fields')
            import warnings
            warnings.warn("Passing 'fields' as keyword argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            fields = None

        if fields is None:
            fields = kwargs.items()
        elif kwargs:
            raise TypeError("Either list of fields or keywords"
                            " can be provided to NamedTuple, not both")
        return _make_nmtuple(typename, fields)
    __new__.__text_signature__ = '($cls, typename, fields=None, /, **kwargs)'


def _dict_new(cls, /, *args, **kwargs):
    return dict(*args, **kwargs)


def _typeddict_new(cls, typename, fields=None, /, *, total=True, **kwargs):
    if fields is None:
        fields = kwargs
    elif kwargs:
        raise TypeError("TypedDict takes either a dict or keyword arguments,"
                        " but not both")

    ns = {'__annotations__': dict(fields), '__total__': total}
    try:
        # Setting correct module is necessary to make typed dict classes pickleable.
        ns['__module__'] = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return _TypedDictMeta(typename, (), ns)


def _check_fails(cls, other):
    # Typed dicts are only for static structural subtyping.
    raise TypeError('TypedDict does not support instance and class checks')


class _TypedDictMeta(type):
    def __new__(cls, name, bases, ns, total=True):
        """Create new typed dict class object.

        This method is called directly when TypedDict is subclassed,
        or via _typeddict_new when TypedDict is instantiated. This way
        TypedDict supports all three syntax forms described in its docstring.
        Subclasses and instances of TypedDict return actual dictionaries
        via _dict_new.
        """
        ns['__new__'] = _typeddict_new if name == 'TypedDict' else _dict_new
        tp_dict = super(_TypedDictMeta, cls).__new__(cls, name, (dict,), ns)

        anns = ns.get('__annotations__', {})
        msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
        anns = {n: _type_check(tp, msg) for n, tp in anns.items()}
        for base in bases:
            anns.update(base.__dict__.get('__annotations__', {}))
        tp_dict.__annotations__ = anns
        if not hasattr(tp_dict, '__total__'):
            tp_dict.__total__ = total
        return tp_dict

    __instancecheck__ = __subclasscheck__ = _check_fails


class TypedDict(dict, metaclass=_TypedDictMeta):
    """A simple typed namespace. At runtime it is equivalent to a plain dict.

    TypedDict creates a dictionary type that expects all of its
    instances to have a certain set of keys, where each key is
    associated with a value of a consistent type. This expectation
    is not checked at runtime but is only enforced by type checkers.
    Usage::

        class Point2D(TypedDict):
            x: int
            y: int
            label: str

        a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
        b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check

        assert Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')

    The type info can be accessed via Point2D.__annotations__. TypedDict
    supports two additional equivalent forms::

        Point2D = TypedDict('Point2D', x=int, y=int, label=str)
        Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

    By default, all keys must be present in a TypedDict. It is possible
    to override this by specifying totality.
    Usage::

        class point2D(TypedDict, total=False):
            x: int
            y: int

    This means that a point2D TypedDict can have any of the keys omitted.A type
    checker is only expected to support a literal False or True as the value of
    the total argument. True is the default, and makes all items defined in the
    class body be required.

    The class syntax is only supported in Python 3.6+, while two other
    syntax forms work for Python 2.7 and 3.2+
    """


def NewType(name, tp):
    """NewType creates simple unique types with almost zero
    runtime overhead. NewType(name, tp) is considered a subtype of tp
    by static type checkers. At runtime, NewType(name, tp) returns
    a dummy function that simply returns its argument. Usage::

        UserId = NewType('UserId', int)

        def name_by_id(user_id: UserId) -> str:
            ...

        UserId('user')          # Fails type check

        name_by_id(42)          # Fails type check
        name_by_id(UserId(42))  # OK

        num = UserId(5) + 1     # type: int
    """

    def new_type(x):
        return x

    new_type.__name__ = name
    new_type.__supertype__ = tp
    return new_type

# Python-version-specific alias (Python 2: unicode; Python 3: str)
Text = str
# Constant that's True when type checking, but False here.
TYPE_CHECKING = False

class IO(Generic[AnyStr]):
    """Generic base class for TextIO and BinaryIO.

    This is an abstract, generic version of the return of open().

    NOTE: This does not distinguish between the different possible
    classes (text vs. binary, read vs. write vs. read/write,
    append-only, unbuffered).  The TextIO and BinaryIO subclasses
    below capture the distinctions between text vs. binary, which is
    pervasive in the interface; however we currently do not offer a
    way to track the other distinctions in the type system.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def mode(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def closed(self) -> bool:
        pass

    @abstractmethod
    def fileno(self) -> int:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def isatty(self) -> bool:
        pass

    @abstractmethod
    def read(self, n: int = -1) -> AnyStr: # type: ignore
        pass

    @abstractmethod
    def readable(self) -> bool:
        pass

    @abstractmethod
    def readline(self, limit: int = -1) -> AnyStr: # type: ignore
        pass

    @abstractmethod
    def readlines(self, hint: int = -1) -> List[AnyStr]: # type: ignore
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int = 0) -> int:
        pass

    @abstractmethod
    def seekable(self) -> bool:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

    @abstractmethod
    def truncate(self, size: int = None) -> int:
        pass

    @abstractmethod
    def writable(self) -> bool:
        pass

    @abstractmethod
    def write(self, s: AnyStr) -> int:# type: ignore
        pass

    @abstractmethod
    def writelines(self, lines: List[AnyStr]) -> None:# type: ignore
        pass

    @abstractmethod
    def __enter__(self) -> 'IO[AnyStr]':
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback) -> None:
        pass


class BinaryIO(IO[bytes]):
    """Typed version of the return of open() in binary mode."""

    __slots__ = ()

    @abstractmethod
    def write(self, s: Union[bytes, bytearray]) -> int: # type: ignore
        pass

    @abstractmethod
    def __enter__(self) -> 'BinaryIO':
        pass


class TextIO(IO[str]):
    """Typed version of the return of open() in text mode."""

    __slots__ = ()

    @property
    @abstractmethod
    def buffer(self) -> BinaryIO:
        pass

    @property
    @abstractmethod
    def encoding(self) -> str:
        pass

    @property
    @abstractmethod
    def errors(self) -> Optional[str]: # type: ignore
        pass

    @property
    @abstractmethod
    def line_buffering(self) -> bool:
        pass

    @property
    @abstractmethod
    def newlines(self) -> Any: # type: ignore
        pass

    @abstractmethod
    def __enter__(self) -> 'TextIO':
        pass


class io:
    """Wrapper namespace for IO generic classes."""

    __all__ = ['IO', 'TextIO', 'BinaryIO']
    IO = IO
    TextIO = TextIO
    BinaryIO = BinaryIO


io.__name__ = __name__ + '.io'
sys.modules[io.__name__] = io

Pattern = _alias(stdlib_re.Pattern, AnyStr)
Match = _alias(stdlib_re.Match, AnyStr)

class re:
    """Wrapper namespace for re type aliases."""

    __all__ = ['Pattern', 'Match']
    Pattern = Pattern
    Match = Match


re.__name__ = __name__ + '.re'
sys.modules[re.__name__] = re



class Attack(ABC):
    """
    Abstract base class for all attack abstract base classes.
    """

    attack_params: List[str] = [] # type: ignore
    # The _estimator_requirements define the requirements an estimator must satisfy to be used as a target for an
    # attack. They should be a tuple of requirements, where each requirement is either a class the estimator must
    # inherit from, or a tuple of classes which define a union, i.e. the estimator must inherit from at least one class
    # in the requirement tuple.
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None # type: ignore

    def __init__(
        self,
        estimator,
        summary_writer: Union[str, bool, SummaryWriter] = False, # type: ignore
    ):
        """
        :param estimator: An estimator.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        """
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        if not self.is_estimator_valid(estimator, self._estimator_requirements):
            raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        self._estimator = estimator
        self._summary_writer_arg = summary_writer
        self._summary_writer: Optional[SummaryWriter] = None # type: ignore

        if isinstance(summary_writer, SummaryWriter):  # pragma: no cover
            self._summary_writer = summary_writer
        elif summary_writer:
            self._summary_writer = SummaryWriterDefault(summary_writer)

        Attack._check_params(self)

    @property
    def estimator(self):
        """The estimator."""
        return self._estimator

    @property
    def summary_writer(self):
        """The summary writer."""
        return self._summary_writer

    @property
    def estimator_requirements(self):
        """The estimator requirements."""
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of attack-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
            else:
                raise ValueError(f'The attribute "{key}" cannot be set for this attack.')
        self._check_params()

    def _check_params(self) -> None:

        if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
            raise ValueError("The argument `summary_writer` has to be either of type bool or str.")

    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        """
        Checks if the given estimator satisfies the requirements for this attack.

        :param estimator: The estimator to check.
        :param estimator_requirements: Estimator requirements.
        :return: True if the estimator is valid for the attack.
        """

        for req in estimator_requirements:
            # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
            # estimator is required to inherit from at least one of the classes
            if isinstance(req, tuple):
                if all(p not in type(estimator).__mro__ for p in req):
                    return False
            elif req not in type(estimator).__mro__:
                return False
        return True

    def __repr__(self):
        """
        Returns a string describing the attack class and attack_params
        """
        param_str = ""
        for param in self.attack_params:
            if hasattr(self, param):
                param_str += f"{param}={getattr(self, param)}, "
            elif hasattr(self, "_attack"):
                if hasattr(self._attack, param):
                    param_str += f"{param}={getattr(self._attack, param)}, "
        return f"{type(self).__name__}({param_str})"


if TYPE_CHECKING:
    from art.utils import OBJECT_DETECTOR_TYPE
logger = logging.getLogger(__name__)



class EvasionAttack(Attack):
    """
    Abstract base class for evasion attack classes.
    """

    def __init__(self, **kwargs) -> None:
        self._targeted = False
        super().__init__(**kwargs)

    @abstractmethod
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray: # type: ignore
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
                  or not. This parameter is only used by some of the attacks.
        :return: An array holding the adversarial examples.
        """
        raise NotImplementedError

    @property
    def targeted(self) -> bool:
        """
        Return Boolean if attack is targeted. Return None if not applicable.
        """
        return self._targeted

    @targeted.setter
    def targeted(self, targeted) -> None:
        self._targeted = targeted


#######################################################################################
##################################Core codes of basical patch method###################
#######################################################################################
class AdvAttenuationPatch(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "patch_location",
        "crop_range",
        "brightness_range",
        "rotation_weights",
        "sample_size",
        "targeted",
        "summary_writer",
        "verbose",
        "flag"
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: Tuple[int, int, int] = (40, 40, 3), # type: ignore
        patch_location: Tuple[int, int] = (0, 0), # type: ignore
        crop_range: Tuple[int, int] = (0, 0), # type: ignore
        brightness_range: Tuple[float, float] = (0.7, 1.5), # type: ignore
        rotation_weights: Union[Tuple[float, float, float, float], Tuple[int, int, int, int]] = (0.7, 0.1, 0.1, 0.1), # type: ignore
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        targeted: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False, # type: ignore
        verbose: bool = True,
        flag = 1
    ):
        """
        Create an instance of the :class:`.AdvAttenuationPatch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (height, width, nb_channels).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param crop_range: By how much the images may be cropped as a tuple of shape (height, width).
        :param brightness_range: Range for randomly adjusting the brightness of the image.
        :param rotation_weights: Sampling weights for random image rotations by (0, 90, 180, 270) degrees
                                 counter-clockwise.
        :param sample_size: Number of samples to be used in expectations over transformation.
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """

        super().__init__(estimator=estimator, summary_writer=summary_writer)

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        if self.estimator.clip_values is None:
            self._patch = np.zeros(shape=patch_shape, dtype=config.ART_NUMPY_DTYPE)
        else:
            self._patch = (
                np.random.randint(0, 255, size=patch_shape)
                / 255
                * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                + self.estimator.clip_values[0]
            ).astype(config.ART_NUMPY_DTYPE)
        self.verbose = verbose
        self.patch_location = patch_location
        self.crop_range = crop_range
        self.brightness_range = brightness_range
        self.rotation_weights = rotation_weights
        self.sample_size = sample_size
        self._targeted = targeted
        self._check_params()
        self.flag = flag

    def generate(  # type: ignore
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]] = None, **kwargs) -> np.ndarray: # type: ignore
        """
        Generate AdvAttenuationPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """
        energy_tau = 0.5

        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if y is None and self.targeted:
            raise ValueError("The targeted version of AdvAttenuationPatch attack requires target labels provided to `y`.")
        if y is not None and not self.targeted:
            raise ValueError("The AdvAttenuationPatch attack does not use target labels.")
        if x.ndim != 4:  # pragma: no cover
            raise ValueError("The adversarial patch can only be applied to images.")

        # Check whether patch fits into the cropped images:
        if self.estimator.channels_first:
            image_height, image_width = x.shape[2:4]
        else:
            image_height, image_width = x.shape[1:3]

        if not self.estimator.native_label_is_pytorch_format and y is not None:
            from art.estimators.object_detection.utils import convert_tf_to_pt
            y = convert_tf_to_pt(y=y, height=x.shape[1], width=x.shape[2])

        if y is not None:
            for i_image in range(x.shape[0]):#reading single image from the whole batch
                y_i = y[i_image]["boxes"]
                for i_box in range(y_i.shape[0]):
                    x_1, y_1, x_2, y_2 = y_i[i_box]
                    if (  # pragma: no cover
                        x_1 < self.crop_range[1]
                        or y_1 < self.crop_range[0]
                        or x_2 > image_width - self.crop_range[1] + 1
                        or y_2 > image_height - self.crop_range[0] + 1
                    ):
                        raise ValueError("Cropping is intersecting with at least one box, reduce `crop_range`.")

        if (  # pragma: no cover patch is located based on the top left corner
            self.patch_location[0] + self.patch_shape[1] > image_height - self.crop_range[0]
            or self.patch_location[1] + self.patch_shape[2] > image_width - self.crop_range[1]
        ):
            raise ValueError("The patch (partially) lies outside the cropped image.")

        for i_step in trange(self.max_iter, desc="AdvAttenuationPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients_old = np.zeros_like(self._patch)

            for e_step in range(self.sample_size):
                if e_step == 0 or (e_step + 1) % 100 == 0:
                    logger.info("EOT Step: %i", e_step + 1)

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size, x.shape[0])

                    if y is None:
                        y_batch = y
                    else:
                        y_batch = y[i_batch_start:i_batch_end]

                    # Sample and apply the random transformations:
                    patched_images, patch_target, transforms = self._augment_images_with_patch(
                        x[i_batch_start:i_batch_end], y_batch, self._patch, channels_first=self.estimator.channels_first
                    )

                    gradients = self.estimator.loss_gradient(
                        x=patched_images,
                        y=patch_target,
                        standardise_output=True,
                    )

                    gradients = self._untransform_gradients(
                        gradients, transforms, channels_first=self.estimator.channels_first
                    )

                    patch_gradients = patch_gradients_old + np.sum(gradients, axis=0)
                    logger.debug(
                        "Gradient percentage diff: %f)",
                        np.mean(np.sign(patch_gradients) != np.sign(patch_gradients_old)),
                    )

                    patch_gradients_old = patch_gradients

            # Write summary
            if self.summary_writer is not None:  # pragma: no cover
                x_patched, y_patched, _ = self._augment_images_with_patch(
                    x, y, self._patch, channels_first=self.estimator.channels_first
                )

                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_step,
                    grad=np.expand_dims(patch_gradients, axis=0),
                    patch=self._patch,
                    estimator=self.estimator,
                    x=x_patched,
                    y=y_patched,
                    targeted=self.targeted,
                )

            # update patch by gradient ascent 
            self._patch = self._patch + np.sign(patch_gradients) * (1 - 2 * int(self.targeted)) * self.learning_rate
            
            #x shape is batch,channel,height,width
            #The range of Clip function is determined by energy coefficient
            x_patch_area = x[:,:,self.patch_location[0]:self.patch_location[0] + self.patch_shape[1],self.patch_location[1]:self.patch_location[1] + self.patch_shape[2]]
            
            raw_area_energy = np.mean(x_patch_area)
            mean = np.mean(self._patch)
            std_dev = np.std(self._patch)

            target_mean = energy_tau*raw_area_energy
            
            patch_adjust = self._patch - mean + target_mean
            clip_min = target_mean - 2 * std_dev
            clip_max = target_mean + 2 * std_dev
            self._patch = np.clip(
                    patch_adjust,
                    a_min=max(clip_min,0),
                    a_max=clip_max,)

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return self._patch

    def _augment_images_with_patch(
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]], patch: np.ndarray, channels_first: bool # type: ignore
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Union[int, float]]]: # type: ignore
        """
        Augment images with patch.

        :param x: Sample images.
        :param y: Target labels.
        :param patch: The patch to be applied.
        :param channels_first: Set channels first or last.
        """

        transformations: Dict[str, Union[float, int]] = {} # type: ignore
        x_copy = x.copy()
        patch_copy = patch.copy()
        x_patch = x.copy()

        if channels_first:
            x_copy = np.transpose(x_copy, (0, 2, 3, 1))
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_copy = np.transpose(patch_copy, (1, 2, 0))

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_copy.shape[0], y_1 + patch_copy.shape[1]
        if(self.flag == False):
            x_patch[:, x_1:x_2, y_1:y_2, :] -= patch_copy
            # Increase way for normal image
        else:
            x_patch[:, x_1:x_2, y_1:y_2, :] -= patch_copy
            #print("func: augment_patch now minus")

        # 1) crop images:
        crop_x: int = random.randint(0, self.crop_range[0])
        crop_y = random.randint(0, self.crop_range[1])
        x_1, y_1 = crop_x, crop_y
        x_2, y_2 = x_copy.shape[1] - crop_x + 1, x_copy.shape[2] - crop_y + 1
        x_copy = x_copy[:, x_1:x_2, y_1:y_2, :]
        x_patch = x_patch[:, x_1:x_2, y_1:y_2, :]

        transformations.update({"crop_x": crop_x, "crop_y": crop_y})

        # 2) rotate images:
        rot90 = random.choices([0, 1, 2, 3], weights=self.rotation_weights)[0]

        x_copy = np.rot90(x_copy, rot90, (1, 2))
        x_patch = np.rot90(x_patch, rot90, (1, 2))

        transformations.update({"rot90": rot90})

        if y is not None:
            y_copy: List[Dict[str, np.ndarray]] = [] # type: ignore
            for i_image in range(x_copy.shape[0]):
                y_b = y[i_image]["boxes"].copy()
                image_width = x.shape[2]
                image_height = x.shape[1]
                x_1_arr = y_b[:, 0]
                y_1_arr = y_b[:, 1]
                x_2_arr = y_b[:, 2]
                y_2_arr = y_b[:, 3]
                box_width = x_2_arr - x_1_arr
                box_height = y_2_arr - y_1_arr

                if rot90 == 0:
                    x_1_new = x_1_arr
                    y_1_new = y_1_arr
                    x_2_new = x_2_arr
                    y_2_new = y_2_arr

                if rot90 == 1:
                    x_1_new = y_1_arr
                    y_1_new = image_width - x_1_arr - box_width
                    x_2_new = y_1_arr + box_height
                    y_2_new = image_width - x_1_arr

                if rot90 == 2:
                    x_1_new = image_width - x_2_arr
                    y_1_new = image_height - y_2_arr
                    x_2_new = x_1_new + box_width
                    y_2_new = y_1_new + box_height

                if rot90 == 3:
                    x_1_new = image_height - y_1_arr - box_height
                    y_1_new = x_1_arr
                    x_2_new = image_height - y_1_arr
                    y_2_new = x_1_arr + box_width

                y_i = {}
                y_i["boxes"] = np.zeros_like(y[i_image]["boxes"])
                y_i["boxes"][:, 0] = x_1_new
                y_i["boxes"][:, 1] = y_1_new
                y_i["boxes"][:, 2] = x_2_new
                y_i["boxes"][:, 3] = y_2_new

                y_i["labels"] = y[i_image]["labels"]
                y_i["scores"] = y[i_image]["scores"]

                y_copy.append(y_i)

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy = np.round(brightness * x_copy / self.learning_rate) * self.learning_rate
        x_patch = np.round(brightness * x_patch / self.learning_rate) * self.learning_rate

        transformations.update({"brightness": brightness})

        logger.debug("Transformations: %s", str(transformations))

        patch_target: List[Dict[str, np.ndarray]] = [] # type: ignore

        if self.targeted:
            predictions = y_copy
        else:
            if channels_first:
                x_copy = np.transpose(x_copy, (0, 3, 1, 2))
            predictions = self.estimator.predict(x=x_copy, standardise_output=True)

        for i_image in range(x_copy.shape[0]):
            target_dict = {}
            target_dict["boxes"] = predictions[i_image]["boxes"]
            target_dict["labels"] = predictions[i_image]["labels"]
            target_dict["scores"] = predictions[i_image]["scores"]
            patch_target.append(target_dict)

        if channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch, patch_target, transformations

    def _untransform_gradients(
        self,
        gradients: np.ndarray,
        transforms: Dict[str, Union[int, float]], # type: ignore
        channels_first: bool,
    ) -> np.ndarray:
        """
        Revert transformation on gradients.

        :param gradients: The gradients to be reverse transformed.
        :param transforms: The transformations in forward direction.
        :param channels_first: Set channels first or last.
        """

        if channels_first:
            gradients = np.transpose(gradients, (0, 2, 3, 1))

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo rotations:
        rot90 = int((4 - transforms["rot90"]) % 4)
        gradients = np.rot90(gradients, k=rot90, axes=(1, 2))

        # Account for cropping when considering the upper left point of the patch:
        x_1 = self.patch_location[0] - int(transforms["crop_x"])
        y_1 = self.patch_location[1] - int(transforms["crop_y"])
        if channels_first:
            x_2 = x_1 + self.patch_shape[1]
            y_2 = y_1 + self.patch_shape[2]
        else:
            x_2 = x_1 + self.patch_shape[0]
            y_2 = y_1 + self.patch_shape[1]
        gradients = gradients[:, x_1:x_2, y_1:y_2, :]

        if channels_first:
            gradients = np.transpose(gradients, (0, 3, 1, 2))

        return gradients

    def apply_patch(self, x: np.ndarray, patch_external: Optional[np.ndarray] = None) -> np.ndarray: # type: ignore
        """
        Apply the adversarial patch to images.
        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :return: The patched images.
        """

        x_patch = x.copy()

        if patch_external is not None:
            patch_local = patch_external.copy()
        else:
            patch_local = self._patch.copy()

        if self.estimator.channels_first:
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_local = np.transpose(patch_local, (1, 2, 0))

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]

        if x_2 > x_patch.shape[1] or y_2 > x_patch.shape[2]:  # pragma: no cover
            raise ValueError("The patch (partially) lies outside the image.")

        # get patch by minus way
        x_patch[:, x_1:x_2, y_1:y_2, :] = np.uint8(np.clip(np.float32(x_patch[:, x_1:x_2, y_1:y_2, :])-np.float32(patch_local),0,255))

        if self.estimator.channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch

    def _check_params(self) -> None:
        if not isinstance(self.patch_shape, (tuple, list)) or not all(isinstance(s, int) for s in self.patch_shape):
            raise ValueError("The patch shape must be either a tuple or list of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if not isinstance(self.patch_location, (tuple, list)) or not all(
            isinstance(s, int) for s in self.patch_location
        ):
            raise ValueError("The patch location must be either a tuple or list of integers.")
        if len(self.patch_location) != 2:
            raise ValueError("The length of patch location must be 2.")

        if not isinstance(self.crop_range, (tuple, list)) or not all(isinstance(s, int) for s in self.crop_range):
            raise ValueError("The crop range must be either a tuple or list of integers.")
        if len(self.crop_range) != 2:
            raise ValueError("The length of crop range must be 2.")

        if self.crop_range[0] > self.crop_range[1]:
            raise ValueError("The first element of the crop range must be less or equal to the second one.")

        if self.patch_location[0] < self.crop_range[0] or self.patch_location[1] < self.crop_range[1]:
            raise ValueError("The patch location must be outside the crop range.")

        if not isinstance(self.brightness_range, (tuple, list)) or not all(
            isinstance(s, float) for s in self.brightness_range
        ):
            raise ValueError("The brightness range must be either a tuple or list of floats.")
        if len(self.brightness_range) != 2:
            raise ValueError("The length of brightness range must be 2.")

        if self.brightness_range[0] < 0.0:
            raise ValueError("The brightness range must be >= 0.0.")

        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("The first element of the brightness range must be less or equal to the second one.")

        if not isinstance(self.rotation_weights, (tuple, list)) or not all(
            isinstance(s, (float, int)) for s in self.rotation_weights
        ):
            raise ValueError("The rotation sampling weights must be provided as tuple or list of float or int values.")
        if len(self.rotation_weights) != 4:
            raise ValueError("The number of rotation sampling weights must be 4.")

        if not all(s >= 0.0 for s in self.rotation_weights):
            raise ValueError("The rotation sampling weights must be non-negative.")

        if all(s == 0.0 for s in self.rotation_weights):
            raise ValueError("At least one of the rotation sampling weights must be strictly greater than zero.")

        if not isinstance(self.sample_size, int):
            raise ValueError("The EOT sample size must be of type int.")
        if self.sample_size <= 0:
            raise ValueError("The EOT sample size must be greater than 0.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")




####################################################################################################################################################
################################################################ Core parts of innovative codes#####################################################
####################################################################################################################################################


"""
#################        Helper functions and labels          #################
"""
COCO_INSTANCE_CATEGORY_NAMES = [
'ships'
]
"""
#################        Evasion and Attack Parameters settings        #################
"""
eps = 32
eps_step = 2
max_iter = 10


weight_path = './weights/shipv5m.pt'
save_path = './adv_examples/attackv1_3/'
src_path = './dataset/clean_val/'
test_img_num = 1000

"""
#################        Model definition        #################
"""
MODEL = "yolov5"  # OR yolov5
if MODEL == "yolov3":
    print("NO! yolov3 not supported!")
elif MODEL == "yolov5":
    from loss import ComputeLoss # the setting of loss come from the exact loss.py
    matplotlib.use("Agg")

    class Yolo(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.hyp = {
                "box": 0.05,
                "obj": 1.0,
                "cls": 0.5,
                "anchor_t": 4.0,
                "cls_pw": 1.0,
                "obj_pw": 1.0,
                "fl_gamma": 0.0,
            }
            self.compute_loss = ComputeLoss(self.model.model.model)

        def forward(self, x, targets=None):
            if self.training:
                outputs = self.model.model.model(x)
                loss, loss_items,loss_C= self.compute_loss(outputs, targets)
                loss_components_dict = {"loss_total": loss_C}
                return loss_components_dict
            else:
                return self.model(x)

    model = yolov5.load(weight_path)
    model = Yolo(model)
    #model.to(device)  # i.e. device=torch.device(0)

    detector = PyTorchYolo(
        model=model, device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
    )

    if hasattr(detector.model, 'amp'):
            detector.model.amp = False


"""
#################        new tools       #################
"""
def plt_show_img(name, img_src):
    plt.figure()
    plt.title(name)
    plt.imshow(img_src)
    plt.show()

def line_row_statics2(img_src):
    h, w = img_src.shape
    #img_src = np.clip(img_src,175,255)
    #img_src[img_src == 175] = 0
    line_statics = np.zeros(w)
    cor_lin = np.arange(w)
    for i in range(w):
        for j in range(h):
            if img_src[i][j]:
                line_statics[i] += 1
    row_statics = np.zeros(h)
    cor_row = np.arange(h)
    for j in range(h):
        for i in range(w):
            if img_src[i][j]:
                row_statics[j] += 1
    x = np.argmax(line_statics)
    y = np.argmax(row_statics)
    #print("line:{}, row:{}".format(x,y))
    lamta = 640/256
    xx = int(x*lamta) - 10
    yy = int(y*lamta) - 10
    return xx,yy

def cv2_ship_seg(img):
    img = np.clip(img,175,255)
    img[img == 175] = 0
    #plt_show_img('1',img)
    blurd_img = cv2.GaussianBlur(img, (7, 7), 0)
    gray_img = cv2.cvtColor(blurd_img, cv2.COLOR_BGR2GRAY)
    #plt_show_img('2',gray_img)
    ret, binary = cv2.threshold(gray_img, 175, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    output = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    labels_exact = np.zeros((5, 256, 256))
    h, w = np.shape(labels)
    # find the useful layer, 2ed
    for i in range(5):
        for j in range(h):
            for k in range(w):
                if labels[j][k] == i:
                    labels_exact[i][j][k] = labels[j][k]
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))
    colors[0] = (0, 0, 0)

    h, w = gray_img.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[labels[row, col]]
    #plt_show_img('final',image)
    #print("total components : ", num_labels - 1)
    return_img = rgb2gray(image)
    return return_img

def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2125, 0.7154, 0.0721])

def get_shape(sg_img,ship_loc,ship_indx):
    sg_img[sg_img < 10 ] = 0
    sg_img[sg_img != 0 ] = 1
    #plt_show_img(' ',sg_img)
    patch_h = ship_loc[ship_indx][0]
    patch_w = ship_loc[ship_indx][1]
    shape = 5
    flg = True
    x_sg = sg_img.copy()
    while(flg):
        ship_size = x_sg.sum()
        cp_img = np.zeros((640,640))
        cp_img[patch_h:patch_h+shape,patch_w:patch_w+shape] = 1
        #plt_show_img(" ",cp_img)
        masked_img  = x_sg.copy()
        masked_img[patch_h:patch_h+shape,patch_w:patch_w+shape] = 0
        masked_img = x_sg - masked_img
        #plt_show_img("masked",masked_img)
        if(masked_img.sum() >= cp_img.sum()*0.95):
            shape+=1
        else:
            if(cp_img.sum()<0.25*ship_size):
                shape = int(math.sqrt(0.25*ship_size)) 
                cp_img[patch_h:patch_h+shape,patch_w:patch_w+shape] = 1
            sg_img = sg_img - cp_img
            flg = False
    return shape

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')] 

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

def data_loader(src): #attack the whole val dataset
    img_list = get_imlist(os.path.join(src, 'images'))
    txt_list = get_txtlist(os.path.join(src, 'labels'))
    txt_lenth: int = len(txt_list)
    loc_list = []
    for i in range(txt_lenth):
        a = txt_list[i]
        with open(a) as fl:
            img_pre_loc = fl.readlines()
        loc_info_list = [line.split() for line in img_pre_loc]
        img_p = txt_list[i].split('/')[-1]
        img_p = src + 'images/' + img_p[:-3] + 'jpg'
        ship_num = len(loc_info_list)
        ship_loc = []
        print(img_p,'\n',loc_info_list,'\n',ship_num)
        for j in range(ship_num):
            mask = np.ones((256,256,3))
            loc_y = int(256*float(loc_info_list[j][1]))
            loc_x = int(256*float(loc_info_list[j][2]))
            loc_h = int(256*float(loc_info_list[j][3]))
            loc_w = int(256*float(loc_info_list[j][4]))
            mask[loc_x-loc_w//2:loc_x+loc_w//2,loc_y-loc_h//2:loc_y+loc_h//2,:] = 0
            img_mid = cv2.imread(img_p)
            ship_mid_raw1 = img_mid.copy()
            ship_mid_raw2 = img_mid.copy()
            ship_mid_raw1[loc_x-loc_w//2:loc_x+loc_w//2,loc_y-loc_h//2:loc_y+loc_h//2,:] = 0
            ship_mid = ship_mid_raw2 -ship_mid_raw1
            segged_ship_mid = cv2_ship_seg(ship_mid)
            x1, y1 = line_row_statics2(segged_ship_mid)
            if([x1,y1] <= [0,0] or x1<=0 or y1<=0):
                x1 = int((640/256)*loc_x)
                y1 = int((640/256)*loc_y)
            ship_loc.append([x1,y1])
        loc_list.append(ship_loc)
    return img_list,txt_list,loc_list

#img_list,txt_list,loc_list = data_loader(src=src_path)

def data_loader_test(src, is_test=False, test_num=None): # attack part of the val dataset
    """
    Two types of information (especially patch loaction info) is returned, from txt label and prejection profile respectively,
    nanmely txt_list_out and loc_list_out
    """
    src_path = Path(src)

    image_dir = src_path / "images"
    label_dir = src_path / "labels"

    txt_files = sorted(list(label_dir.glob("*.txt")))

    if is_test and test_num:
        txt_files = txt_files[:test_num]

    img_list_out = []
    txt_list_out = []
    loc_list_out = []

    pbar = tqdm(txt_files, desc="Loading Data")
    
    for txt_path in pbar:
        img_path = image_dir / f"{txt_path.stem}.jpg"

        if not img_path.exists():
            print(f"\n[Warning] image not found: {img_path}")
            continue
        img_p_str = img_path.as_posix()
        txt_p_str = txt_path.as_posix()

        with open(txt_path, 'r') as f:
            loc_info_list = [line.split() for line in f.readlines()]

        ship_num = len(loc_info_list)
        ship_loc = []
        #print(img_p,'\n',loc_info_list,'\n',ship_num)

        img_mid_raw = cv2.imread(img_p_str)
        if img_mid_raw is None:
            continue

        for j in range(ship_num):
            loc_y = int(256 * float(loc_info_list[j][1]))
            loc_x = int(256 * float(loc_info_list[j][2]))
            loc_h = int(256 * float(loc_info_list[j][3]))
            loc_w = int(256 * float(loc_info_list[j][4]))

            y_start, y_end = max(0, loc_y - loc_h // 2), min(256, loc_y + loc_h // 2)
            x_start, x_end = max(0, loc_x - loc_w // 2), min(256, loc_x + loc_w // 2)
            ship_mid_raw1 = img_mid_raw.copy()

            ship_mid_raw1[x_start:x_end, y_start:y_end, :] = 0
            ship_mid = img_mid_raw - ship_mid_raw1
        
            segged_ship_mid = cv2_ship_seg(ship_mid)
            x1, y1 = line_row_statics2(segged_ship_mid)
            
            if x1 <= 0 or y1 <= 0:
                x1 = int((640 / 256) * loc_x)
                y1 = int((640 / 256) * loc_y)
            
            ship_loc.append([x1, y1])
        img_list_out.append(img_p_str)
        txt_list_out.append(txt_p_str)
        loc_list_out.append(ship_loc)
    return img_list_out, txt_list_out, loc_list_out


"""
#######################################################################################################################
#############################################      attack images      #################################################
#######################################################################################################################
"""
def generate_adversarial_image_set(source_path,save_path,iterations=300):
    src_path = Path(source_path)
    ad_path = Path(save_path)

    ad_path.mkdir(parents=True, exist_ok=True)

    img_list,txt_list,loc_list = data_loader_test(source_path, True, test_img_num) #The loc_list stores the geometric center points of the ships after the images have been scaled down to 640.
    
    for i in tqdm(range(test_img_num)):
        txt_path = Path(txt_list[i])
        file_stem = txt_path.stem

        img_file_path = src_path / "images" / f"{file_stem}.jpg"
        img_save_path = ad_path / f"patched_{file_stem}.jpg"

        img_name = source_path + 'images/' +(txt_list[i].split('/')[-1])[:-3] + 'jpg'
        img_save_name = 'patched_' + (txt_list[i].split('/')[-1])[:-3] + 'jpg'
        
        img_raw = cv2.imread(img_file_path.as_posix())
        img = cv2.imread(img_file_path.as_posix())
        if img_raw is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        img = cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR) #Image scaling interpolation method utilized by yolov5
        bright_flag = bright_judge(img_name)

        img_reshape = img.transpose((2, 0, 1)) #channel,width,height

        image = np.stack([img_reshape], axis=0).astype(np.float32) #convert into batch style

        a = txt_list[i]
        with open(a) as fl:
            img_pre_loc = fl.readlines()
        loc_info_list = [line.split() for line in img_pre_loc]
        ship_loc_list = loc_list[i]
        patch_num = len(ship_loc_list)
        patted_img = np.zeros((640,640,3)).astype(np.uint8)
        x2 = image.copy()

        for k in range(patch_num):
            loc_y = int(256*float(loc_info_list[k][1]))
            loc_x = int(256*float(loc_info_list[k][2]))
            loc_h = int(256*float(loc_info_list[k][3]))
            loc_w = int(256*float(loc_info_list[k][4]))

            original_height, original_width, _ = img_raw.shape

            new_width,new_height,_ = img.shape

            # calculate the proportion 
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height

            # adjust loc_x, loc_y, loc_w, loc_h
            loc_w_scaled = int(loc_w * width_ratio)
            loc_h_scaled = int(loc_h * height_ratio)    

            new_x = int(loc_x * width_ratio - 0.5*loc_w * width_ratio) 
            new_y = int(loc_y * height_ratio - 0.5*loc_h * height_ratio)

            if(new_x > 640 - loc_w_scaled):
                new_x = 640 - loc_w_scaled
            if(new_y > 640 - loc_h_scaled):
                new_y = 640 - loc_h_scaled
            if(new_x < 0):
                new_x = 0
            if(new_y < 0):
                new_y = 0

            attack = AdvAttenuationPatch(estimator=detector, patch_shape= (3,loc_w_scaled,loc_h_scaled),patch_location=(new_x,new_y), learning_rate= 5.0, max_iter = iterations, batch_size= 16, verbose= True,flag = bright_flag)
            image_adv = attack.generate(x=x2, y=None)
            image_adv[:,:,:]=image_adv[0,:,:]*0.299+image_adv[1,:,:]*0.587+image_adv[2,:,:]*0.114
            image_adv=attack.apply_patch(x=x2,patch_external=image_adv)
           
            x2 = image_adv
            patted_img = x2[0].transpose(1, 2, 0)
        cv2.imwrite(str(ad_path / img_save_name),patted_img)
        #print("save path is:",ad_path+img_save_name)

generate_adversarial_image_set(src_path,save_path,iterations=100)
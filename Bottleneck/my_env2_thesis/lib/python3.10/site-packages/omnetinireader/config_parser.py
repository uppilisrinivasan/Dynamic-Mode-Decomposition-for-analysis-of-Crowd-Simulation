from __future__ import annotations
from calendar import c
from cmath import exp
import copy
import enum
from collections.abc import MutableMapping
from configparser import ConfigParser, NoOptionError
from json.decoder import JSONDecodeError
import os
import logging
import re
import json
import hjson
from typing import List

class CfgValue:

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return self.__str__()

class UnitValue(CfgValue):

    @classmethod
    def s(cls, value):
        return cls(value, "s")
    
    @classmethod
    def m(cls, value):
        return cls(value, "m")
    
    def __init__(self, value, unit) -> None:
        self.value = value
        self.unit = unit

    def __str__(self) -> str:
        return f"{self.value}{self.unit}"

class BoolValue(CfgValue):
    
    def __init__(self, value) -> None:
        self.value = value
    
    def __str__(self) -> str:
        if self.value:
            return "true"
        else:
            return "false"

BoolValue.TRUE =  BoolValue(True)
BoolValue.FALSE = BoolValue(False)

class QString(CfgValue):
    def __init__(self, value) -> None:
        self.value = value
    
    def __str__(self):
        return f'"{self.value}"'

class ObjectValue(CfgValue):

    def __eq__(self, other):
        if isinstance(other, ObjectValue):
            return self._name == other._name and self._value == other._value
        elif isinstance(other, str):
            return self.__str__() == other
        return False

    @staticmethod
    def parse_value_str(str_val) -> dict:
        try:
            val_ = json.loads(str_val)
        except JSONDecodeError:
            try:
                # todo: hjson expects at most one none quoted 
                # string on each line...
                val_ = dict(hjson.loads(str_val.replace(",", "\n").replace("}", "\n}")))
            except Exception as e:
                raise ValueError(f"Error parsing '{str_val}' with json or hjson. Got: {e}")
        return val_

    @classmethod
    def fromString(cls, str_val):
        """
        Creates a ObjectValue object from a given string
        string format must be '{object_name}{object}'
        Example: 'X{"key", "value"}'
        """
        obj_val = cls.parse_string(str_val)
        val = cls.parse_value_str(obj_val[1])
        return cls(name=obj_val[0], value=val)

    @classmethod
    def from_args(cls, name, *args):
        values = {}
        if (len(args) % 2) != 0:
            raise ValueError("expected even number of args")
        for k, v in zip(args[0::2], args[1::2]):
            values[k] = v
        
        return cls(name, values)

    
    def copy(self, *args):
        _copy = ObjectValue(self._name, dict(self._value))
        if (len(args) % 2) != 0:
            raise ValueError("expected even number of args")
        for k, v in zip(args[0::2], args[1::2]):
            _copy[k] = v
        
        return _copy
        

    def parse_string(str_val: str) -> List[str]:
        """
        Parses object value string
        returns: List[str]
            [0] object_name
            [1] object_string (json)
        """
        try:
            obj_val_rgx = r"(\w*)([{].*?[}])$"
            result = re.search(obj_val_rgx, str_val)
            object_name = result.group(1)
            object_string = result.group(2)
            return [object_name, object_string]
        except AttributeError:
            raise AttributeError(f"Cant find an object in the stirng: {str_val}")

    @classmethod
    def is_object_value(cls, str_val) -> bool:
        """
        Checks if a given str_val is an object value
        """
        try:
            result = cls.parse_string(str_val)
            return bool(result[1])
        except:
            return False

    def __init__(self, name="", value: str|dict={}) -> None:
        """
        string representation of the object. Value must be parsable to a valid json object {}
        It must containing the surrounding curly brackets. Value cannot be an array, only an object.
        Example: {key1: 42, key3: "Foo"}
        """
        self._name = name
        if isinstance(value, str):
            self._value = self.parse_value_str(value)
        else:
            self._value = value

    def __str__(self) -> str:
        _v = []
        for k, v in self._value.items():
            _v.append(f"{k}: {v}")
        _v = "{"  + ", ".join(_v) + "}"
        return f"{self._name}{_v}"

    @property
    def as_dict(self):
        return self._value

    @as_dict.setter
    def as_dict(self, val):
        self._value = val
    
    @property
    def name(self):
        return self._name

    def add(self, key, value, override=True):
        """
        add key/value pair to self._value depending on override flag.
        If override False and value exist return false otherwise true. No error
        """
        if override:
            self._value[key] = value
            return True
        else:
            if key in self.as_dict:
                return False
            else:
                self._value[key] = value
                return True
    
    def __setitem__(self, key, value):
        self._value[key] = value
    
    def __getitem__(self, key):
        return self._value[key]
    
    def __contains__(self, key):
        return key in self._value



class OppParser(ConfigParser):

    def __init__(self, **kwargs):
        kwargs.setdefault("interpolation", None)
        kwargs.setdefault("inline_comment_prefixes", "#")
        super().__init__(**kwargs)

    def optionxform(self, optionstr):
        return optionstr

    @classmethod
    def resolve_includes(cls, ini_path, output_path=None, encoding="utf-8"):
        """
        Read and resolve include directives in ini_path and return string representation.
        This will work on the string representation only and wil thus keep comment intact.
        If output_path is given save result as new file
        """
        opp = cls()
        file_content = opp.create_temp_file_with_includes([ini_path])[0]

        if output_path is None:
            return file_content

        # check if output_path is dir and add file name if it is.
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path,
                                       os.path.basename(ini_path))

        with open(output_path, "w") as f:
            f.writelines(file_content)
        return file_content

    def read(self, filenames, encoding=None):

        if isinstance(filenames, (str, bytes, os.PathLike)):
            filenames = [filenames]

        file_contents = self.create_temp_file_with_includes(filenames)

        if len(file_contents) > 1:
            raise ValueError("Only one *.ini file allowed.")

        # TODO allow multiple files
        read_ok = super().read_string(file_contents[0])
        return read_ok

    def create_temp_file_with_includes(self, filenames):

        file_contents = list()

        for f in filenames:
            self.ini_root_path = os.path.dirname(os.path.abspath(f))
            file_content = self.get_file_content_recursively(f)
            file_contents.append("".join(file_content))

        return file_contents

    def get_lines_include(self, input: str|List[str]):

        line_nrs = {}

        if isinstance(input, str):
            with open(input) as f:
                lines = f.readlines()
        else:
            lines = input

        for idx, line in enumerate(lines):
            if line.strip().startswith("include "):
                _inc_path = line.strip().split(" ")[-1]
                line_nrs[idx] = _inc_path

        return line_nrs

    def has_file_include(self, filename):

        line_nrs = self.get_lines_include(filename)
        if  line_nrs == {}:
            return False
        else:
            return True

    def write_sections(self, fp, sections, space_around_delimiters=True):
        """Write an .ini-format representation of the configuration state.

        If `space_around_delimiters' is True (the default), delimiters
        between keys and values are surrounded by spaces.
        """
        if space_around_delimiters:
            d = " {} ".format(self._delimiters[0])
        else:
            d = self._delimiters[0]

        for section in sections:
            self._write_section(fp, section,
                                self._sections[section].items(), d)

    def get_file_content_recursively(self, f_):

        logging.info(f" Relative path found: {f_}")
        f = os.path.join(os.path.abspath(f_))
        logging.info(f"Generate absolute path: {f_} ")

        current_dir = os.path.dirname(f)

        with open(f) as f_:
            file_content = f_.readlines()

        lines: List[str] = self.get_lines_include(f)

        line_offset = 0
        for line, path in lines.items():
            
            if not os.path.isabs(path):
                path = os.path.join(current_dir, path)
            
            print(f"read file {path}")

            lines_to_include = self.get_file_content_recursively(path)
            _start = line + line_offset
            file_content[_start] = f"###include {file_content[_start]}\n"
            _lines = []
            _lines.extend(file_content[0:_start+1])
            _lines.extend(lines_to_include)
            _lines.append(f"{file_content[_start].strip()} END\n")
            _lines.extend(file_content[_start+1:])
            file_content = _lines
            line_offset = line_offset + len(lines_to_include) + 1

        return file_content



class OppConfigType():
    """
    Set type on OppConfigFileBase to create read-only configurations if needed.
    """

    # READ_ONLY = 1
    # EDIT_LOCAL = 2
    # EXT_DEL_LOCAL = 3

    _read_only = 1 << 0
    _edit_local = 1 << 1
    _allow_deletion = 1 << 2
    _edit_global = 1 << 3
    _ext_allow_del_global = 1 << 4


    def __init__(self, val) -> None:
        self.val = val
    
    def __or__(self, v):
        return self.val | v
    
    def __and__(self, v):
        return self.val & v
    
    def __xor__(self, v):
        return self.val ^ v

    @property
    def read_only(self):
        return (self & self._read_only) != 0
    
    @property
    def edit_local(self):
        return not self.read_only and ( self.edit_global or (self & self._edit_local) !=0)
    
    @property
    def edit_global(self):
        return not self.read_only and (self & self._edit_global) != 0
    
    @property
    def allow_edit(self):
        return self.edit_global or self.edit_local
    
    @property
    def allow_delete(self):
        return (self.edit_local or self.edit_global ) and (self & self._allow_deletion) != 0
    


OppConfigType.READ_ONLY = OppConfigType(OppConfigType._read_only)
    
OppConfigType.EDIT_LOCAL = OppConfigType(OppConfigType._edit_local)
OppConfigType.EDIT_DEL_LOCAL = OppConfigType(OppConfigType.EDIT_LOCAL | OppConfigType._allow_deletion)

OppConfigType.EDIT_GLOBAL = OppConfigType(OppConfigType.EDIT_LOCAL | OppConfigType._edit_global)
OppConfigType.EDIT_DEL_GLOBAL = OppConfigType(OppConfigType.EDIT_GLOBAL | OppConfigType._allow_deletion )



class OppConfigFileBase(MutableMapping):
    """
    Represents an omnetpp.ini file. The extends logic is defined in SimulationManual.pdf p.282 ff.
    Each OppConfigFileBase object has a reference to complete omnetpp.ini configuration file but
    only access's its own options, as well as all options reachable by the search path build
    using the 'extends' option.

    Example(taken form [1]):
    The search path for options for the configuration `SlottedAloha2b` is:
    SlottedAloha2b->SlottedAloha2->SlottedAlohaBase->HighTrafficSettings->General
    ```
    [General]
    ...
    [Config SlottedAlohaBase]
    ...
    [Config LowTrafficSettings]
    ...
    [Config HighTrafficSettings]
    ...

    [Config SlottedAloha1]
    extends = SlottedAlohaBase, LowTrafficSettings
    ...
    [Config SlottedAloha2]
    extends = SlottedAlohaBase, HighTrafficSettings
    ...
    [Config SlottedAloha2a]
    extends = SlottedAloha2
    ...
    [Config SlottedAloha2b]
    extends = SlottedAloha2
    ```
    [1]: https://doc.omnetpp.org/omnetpp/manual/#sec:config-sim:section-inheritance
    """

    @classmethod
    def from_path(
            cls, ini_path, config, cfg_type=OppConfigType.EDIT_LOCAL, is_parent=False
    ):
        _root = OppParser()
        _root.read(ini_path)
        _base_dir = os.path.dirname(ini_path)

        return cls(_root, config, _base_dir, cfg_type, is_parent)

    def __init__(
            self,
            root_cfg: OppParser,
            config_name: str,
            _base_dir: str,
            cfg_type=OppConfigType.EDIT_LOCAL,
            is_parent=False,
    ):
        self._root: OppParser = root_cfg
        self._cfg_type: OppConfigType = cfg_type
        self._sec = self._ensure_config_prefix(config_name)
        self._sec_raw = config_name
        self._base_dir = _base_dir

        if not self._has_section_(self._sec):
            raise ValueError(f"no section found with name {self._sec}")
        self._parent_cfg = []
        self._is_parent = is_parent
        self._section_hierarchy = [self._ensure_config_prefix(self._sec)]
        self.update_selection_hierarchy()


    def update_selection_hierarchy(self):
        self._parent_cfg = []
        self._section_hierarchy = [self._ensure_config_prefix(self._sec)]
        if not self._is_parent:
            stack = [iter(self.parents)]
            while stack:
                for p in stack[0]:
                    if p == "":
                        continue
                    _pp = OppConfigFileBase(self._root, p, self._base_dir, is_parent=True)
                    self._parent_cfg.append(_pp)
                    self._section_hierarchy.append(self._ensure_config_prefix(p))
                    if len(_pp.parents) > 0:
                        stack.append(iter(_pp.parents))
                else:
                    stack.pop(0)
        if self._sec_raw != "General" and self._has_section_("General"):
            self._parent_cfg.append(
                OppConfigFileBase(self._root, "General", self._base_dir, is_parent=True)
            )
            self.section_hierarchy.append("General")

    @property
    def base_path(self):
        return self._base_dir

    @property
    def all_sections(self):
        return self._root.sections()

    @property
    def section_hierarchy(self):
        return self._section_hierarchy

    def resolve_path(self, key, all:bool = False):
        p = re.compile('absFilePath\("(.*)"\)')
        def check(m):
            m = p.match(val)
            if m:
                return os.path.join(self.base_path, m.group(1))
            else:
                return os.path.join(self.base_path, val.strip('"'))

        if all :
            ret = []
            for val in self.get_all(key, with_section=False):
                ret.append(check(val))
            return ret
        else:
            val = self.__getitem__(key)
            return check(val)

    def read(self):
        self.read()
        pass

    def writer(self, fp, selected_config_only=False):
        """ write the current state to the given file descriptor. Caller must close file."""
        if selected_config_only:
            # write current section hierarchy with General first (reverse list)
            self._root.write_sections(fp, self.section_hierarchy[::-1])
        else:
            self._root.write(fp)


    @staticmethod
    def _ensure_config_prefix(val):
        """ All omnetpp configurations start with 'Config'. Add 'Config' if it is missing.  """
        if not val.startswith("Config") and val != "General":
            return f"Config {val.strip()}"
        return val

    @property
    def section(self):
        """ Section managed by this OppConfigFileBase object (read-only) """
        return self._sec

    @property
    def parents(self):
        """ local parents i.e all configurations listed in the 'extends' option (read-only) """
        return [
            s.strip()
            for s in self._getitem_local_("extends", default="").strip().split(",")
        ]

    @property
    def type(self):
        return self._cfg_type

    def is_local(self, option):
        """
        Returns True if the given object belongs directly to the current section and False if
        options is contained higher up the hierarchy OR does not exist.
        """
        return self._contains_local_(option)

    def get_config_for_option(self, option):
        """
        Returns the name of the section the option first occurs search order: local --> general
        or None if option does not exist
        """
        if self._contains_local_(option):
            return self.section
        else:
            for p in self._parent_cfg:
                if p._contains_local_(option):
                    return p.get_config_for_option(option)
        return None

    def _is_json_list(self, _str) -> bool:
        # todo: replace with is object value
        obj_rgx = r"[\[].*?[\]]$"
        return bool(re.search(obj_rgx, str(_str)))

    def _has_section_(self, sec):
        """
        True if section exist in the configuration. Note: Returns also True even if given section is not
        in the section_hierarchy if the current section.
        """
        return self._root.has_section(sec)

    def _getitem_local_(self, k, default=None):
        """
        Search for key in local configuration
        """
        try:
            item = self._root.get(self._sec, k)
            if "\n" in item:
                item = item.replace("\n", " ")
            if ObjectValue.is_object_value(item):
                return ObjectValue().fromString(item)
            if self._is_json_list(item):
                return json.loads(item)
            else:
                return self._root.get(self._sec, k)
        except NoOptionError:
            if default is not None:
                return default
            else:
                raise KeyError(f"key not found. Key: {k}")
        except KeyError:
            return self._root.get(self._sec, k)
        except json.JSONDecodeError as e:
            raise KeyError(f"key '{k}' has invalid json value: '{item}' ")

    def _set_local(self, k, v):
        """
        Set new value for key. OppConfigType checks already done
        """
        self._root.set(self._sec, k, v)

    def _contains_local_(self, k):
        """
        True if key exist in current section (parents are not searched) otherwise False
        """
        return self._root.has_option(self._sec, k)

    def _contained_by_parent(self, k):
        """
        True if key exists in any parent. Note key my exist multiple time but only first occurrence of key
        will be returned. See search path.
        """
        return any([k in parent for parent in self._parent_cfg])

    def _delitem_local(self, k):
        """
        Delete local key.
        """
        self._root.remove_option(self._sec, k)

    def __setitem__(self, k, v):
        if self._cfg_type.read_only:
            raise NotImplementedError("Cannot set value on read only config")

        if self._contains_local_(k):
            self._set_local(k, v)
        elif self._contained_by_parent(k):
            if not self._cfg_type.edit_global:
                raise NotImplementedError("Cannot edit value of parent config")
            else:
                for p in self._parent_cfg:
                    if p._contains_local_(k):
                        p._set_local(k, v)
                        return
        else:
            # todo add option 
            self._set_local(k, v)
            # raise KeyError(f"key not found and object config does not allow addition of new keys. Key: {k}")
        if k == "extends":
            self.update_selection_hierarchy()

    def __delitem__(self, k):
        if not self._cfg_type.allow_delete:
            raise ValueError(
                f"current object does not allow deletion. cfg_type={self._cfg_type}"
            )
        if k not in self:
            raise KeyError(f"key not found. Key: {k}")
        if self._contains_local_(k):
            self._root.remove_option(self._sec, k)
        else:
            raise NotImplementedError(
                f"deletion of parent config option not implemented"
            )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(f"key not found. Key: {k}")

        if self._contains_local_(k):
            return self._getitem_local_(k)
        else:
            for parent in self._parent_cfg:
                try:
                    return parent._getitem_local_(k)
                except KeyError:
                    pass
        raise KeyError(f"key not found. Key: {k}")

    def __contains__(self, k) -> bool:
        if self._contains_local_(k):
            return True
        elif any([k in parent for parent in self._parent_cfg]):
            return True
        else:
            return False

    def __len__(self) -> int:
        _len = 0
        for s in self._section_hierarchy:
            _len += len(self._root.items(s))
        return _len

    def __iter__(self):
        for s in self._section_hierarchy:
            for item in self._root.items(s):
                yield item

    def items(self):
        return list(self.__iter__())

    def keys(self):
        return [k for k, _ in self.__iter__()]

    def values(self):
        return [v for _, v in self.__iter__()]

    def get(self, k, default=None):
        if k in self:
            return self[k]
        else:
            return default
    
    def get_all(self, k, with_section: bool = True):
        ret = {}
        for _sec, _data in self._root._sections.items():
            if k in _data:
                ret[_sec] = _data[k]
        
        if with_section:
            return ret
        else:
            return list(ret.values())

    def setdefault(self, k, default=...):
        if k in self:
            return self[k]
        else:
            if self._cfg_type.read_only:
                raise NotImplementedError(
                    "Object is read only"
                )
            else:
                self._set_local(k, default)
        return default

def updateConfigFile(cfg: OppConfigFileBase, changes: dict, deepcopy: bool = True):
    if deepcopy:
        _c = copy.deepcopy(cfg)
    else:
        _c = cfg
    if "extends" in changes:
        _c["extends"] = changes["extends"]
    for k, v in changes.items():
        if k == "extends":
            continue
        if isinstance(v, CfgValue):
            _c[k] = str(v)
        else:
            _c[k] = v

    return _c


class OppConfigFile(OppConfigFileBase):
    """
    Helpers to manage OMNeT++ specifics not part of the standard ini-Configuration
    * Read/Write int and doubles
    * specify units (i.e. s, dBm, m)
    * Handle string quotes (are part of the value)
    todo: implement
    """

    def __init__(self, root_cfg: OppParser, config_name: str):
        super().__init__(root_cfg, config_name)



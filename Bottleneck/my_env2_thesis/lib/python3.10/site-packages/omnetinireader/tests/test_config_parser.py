import ast
import json
import os
import unittest
import logging
from typing import List, Dict

from omnetinireader.config_parser import OppConfigFile, OppConfigFileBase, OppConfigType, ObjectValue


class OppConfigFileBaseTest(unittest.TestCase):
    file_path = "dir1"

    logging.basicConfig(level=logging.DEBUG)

    DEFAULT_FILE = os.path.join(file_path, "omnetpp.ini")
    OVERRIDE_FILE = os.path.join(file_path, "omnetpp_override.ini")
    NEW_FILE = os.path.join(file_path, "omnetpp_2.ini")
    NEW_FILE_COMP = os.path.join(file_path, "omnetpp_compare.ini")
    NEW_FILE_DEL = os.path.join(file_path, "omnetpp_del.ini")
    NEW_FILE_INC_OTHER = os.path.join(file_path, "omnetpp_include_other.ini")
    ov_name = 'X'
    ov_value = {"key_1": [1], "key_2": [42]}
    ov_value_str = "{'key_1': [1], 'key_2': [42]}"
    ov_str = "X{key_1: [1], key_2: [42]}"
    ov = ObjectValue(name=ov_name, value=ov_value)

    @staticmethod
    def get_object(
            config, cfg_type=OppConfigType.EDIT_LOCAL, is_parent=False, path=DEFAULT_FILE
    ):
        return OppConfigFileBase.from_path(
            os.path.join(os.path.split(__file__)[0], path), config, cfg_type, is_parent
        )

    @staticmethod
    def save_object(obj: OppConfigFileBase, path):
        with open(os.path.join(os.path.split(__file__)[0], path), "w") as f:
            obj.writer(f)

    @staticmethod
    def get_lines(path):
        with open(os.path.join(os.path.split(__file__)[0], path), "r") as f:
            return f.readlines()

    def tearDown(self) -> None:
        f = os.path.join(os.path.split(__file__)[0], self.NEW_FILE)
        if os.path.exists(f):
            os.remove(f)

    def test_set_default_exits(self):
        opp = self.get_object("HighTrafficSettings", OppConfigType.READ_ONLY)
        # set default on exiting must work
        self.assertEqual(opp.setdefault("opt_3", '"val_3"'), '"val_3"')
        # set new will ignore default on existing
        self.assertEqual(opp.setdefault("opt_3", '"new"'), '"val_3"')

    def test_set_default_not_exits(self):
        opp = self.get_object("HighTrafficSettings", OppConfigType.READ_ONLY)
        # must raise error on OppConfigType.READ_ONLY
        self.assertRaises(NotImplementedError, opp.setdefault, "new_key", "42")
        opp = self.get_object("HighTrafficSettings", OppConfigType.EDIT_LOCAL)
        # must work with OppConfigType.EDIT_LOCAL
        self.assertEqual(opp.setdefault("new_key", "42"), "42")
        # new key must be local
        self.assertTrue(opp.is_local("new_key"))
        
        opp = self.get_object("HighTrafficSettings", OppConfigType.EDIT_GLOBAL)
        # must work on OppConfigType.EDIT_GLOBAL
        ret = opp.setdefault("new_key", "42")
        self.assertEqual(ret, "42")
        self.assertEqual(opp["new_key"], "42")
        # new key must be local
        self.assertTrue(opp.is_local("new_key"))

    def test_read_only(self):
        opp = self.get_object("HighTrafficSettings", OppConfigType.READ_ONLY)
        # reading must work
        self.assertEqual(opp["opt_3"], '"val_3"')
        self.assertEqual(opp["general_option"], '"VAL1"')
        # setting new values must not work for (local and parent options)
        self.assertRaises(NotImplementedError, opp.__setitem__, "opt_3", '"new_val"')
        self.assertRaises(
            NotImplementedError, opp.__setitem__, "general_option", '"new_val"'
        )

    def test_edit_local(self):
        opp = self.get_object("HighTrafficSettings", OppConfigType.EDIT_LOCAL)
        # reading must work
        self.assertEqual(opp["opt_3"], '"val_3"')
        self.assertEqual(opp["general_option"], '"VAL1"')
        # setting new values must work for local options only
        opp["opt_3"] = '"new_val"'
        self.assertEqual(opp["opt_3"], '"new_val"')
        # general_option belongs to parent config 'General'
        self.assertRaises(
            NotImplementedError, opp.__setitem__, "general_option", '"new_val"'
        )

    def test_hierarchy(self):
        """ Ensure correct lookup order for extended configurations"""
        opp = self.get_object("SlottedAloha2b", OppConfigType.EDIT_LOCAL)
        self.assertListEqual(
            opp.section_hierarchy,
            [
                "Config SlottedAloha2b",
                "Config SlottedAloha2",
                "Config SlottedAlohaBase",
                "Config HighTrafficSettings",
                "General",
            ],
        )
        opp = self.get_object("SlottedAloha1", OppConfigType.EDIT_LOCAL)
        self.assertListEqual(
            opp.section_hierarchy,
            [
                "Config SlottedAloha1",
                "Config SlottedAlohaBase",
                "Config LowTrafficSettings",
                "General",
            ],
        )
        opp = self.get_object("General", OppConfigType.EDIT_LOCAL)
        self.assertListEqual(opp.section_hierarchy, ["General"])
        opp = self.get_object("SlottedAlohaBase", OppConfigType.EDIT_LOCAL)
        self.assertListEqual(
            opp.section_hierarchy, ["Config SlottedAlohaBase", "General"]
        )
    
    def test_hierarchy_update(self):
        opp = self.get_object("SlottedAloha2b", OppConfigType.EDIT_LOCAL)
        self.assertListEqual(
            opp.section_hierarchy,
            [
                "Config SlottedAloha2b",
                "Config SlottedAloha2",
                "Config SlottedAlohaBase",
                "Config HighTrafficSettings",
                "General",
            ],
        )
        opp["extends"] = "SlottedAlohaBase"
        self.assertListEqual(
            opp.section_hierarchy,
            [
                "Config SlottedAloha2b",
                "Config SlottedAlohaBase",
                "General",
            ],
        )

    def test_override(self):
        opp = self.get_object("SlottedAloha2", OppConfigType.EDIT_GLOBAL)
        opp["opt_5"] = '"val_55"'
        self.save_object(opp, self.NEW_FILE)
        opp2 = opp = self.get_object(
            "SlottedAloha2", OppConfigType.EDIT_GLOBAL, path=self.NEW_FILE
        )
        self.assertEqual(opp2["opt_5"], '"val_55"')
        lines_new = [
            line.replace("\t", " ") for line in self.get_lines(self.NEW_FILE) if not line.startswith("\n")
        ]
        lines_comp = [
            line
            for line in self.get_lines(self.NEW_FILE_COMP)
            if not line.startswith("\n")
        ]
        self.assertListEqual(lines_new, lines_comp)
    
    def test_get_overrides(self):
        opp = self.get_object("Bar", OppConfigType.READ_ONLY, path=self.OVERRIDE_FILE)
        self.assertDictEqual(
            opp.get_all("opt_6"),
            {
                'Config Foo': '"val_6"',
                'Config Bar': '"val_7"',
                'Config Baz': '"val_15"'
            }
        )
        data = opp.resolve_path("opt_6", all=True)
        print("foo")

    def test_safe_to_file(self):
        sa_2 = self.get_object("SlottedAloha2", OppConfigType.EDIT_DEL_GLOBAL)
        hts = self.get_object("HighTrafficSettings", OppConfigType.EDIT_DEL_GLOBAL)
        self.assertEqual(sa_2["opt_HT"], '"overwritten_val_HT"')
        self.assertEqual(
            sa_2["general_option"],
            '"general_option option overwritten by SlottedAloha2"',
        )
        self.assertEqual(hts["opt_HT"], '"val_HT"')
        self.assertEqual(hts["general_option"], '"VAL1"')

    def test_delete_key(self):
        opp = self.get_object("SlottedAloha2", OppConfigType.EDIT_DEL_GLOBAL)
        self.assertEqual(
            opp["general_option"],
            '"general_option option overwritten by SlottedAloha2"',
        )
        del opp["general_option"]
        # after deletion value from General section must be accessible.
        self.assertEqual(opp["general_option"], '"VAL1"')
        self.save_object(opp, self.NEW_FILE)
        lines_new = [
            line.replace("\t", " ") for line in self.get_lines(self.NEW_FILE) if not line.startswith("\n")
        ]
        lines_del = [
            line
            for line in self.get_lines(self.NEW_FILE_DEL)
            if not line.startswith("\n")
        ]
        self.assertListEqual(lines_new, lines_del)

    def test_include_config_(self):
        logging.info(f"################### Read file: {self.NEW_FILE_INC_OTHER}#############################")

        opp = self.get_object(
            "General",
            OppConfigType.READ_ONLY,
            path=self.NEW_FILE_INC_OTHER,
        )
        self.assertEqual(opp['general_option'], '"VAL1"')

        opp = self.get_object(
            "inc_1",
            OppConfigType.READ_ONLY,
            path=self.NEW_FILE_INC_OTHER,
        )
        self.assertEqual(opp['*.node[*].numApps'], '2')

        opp = self.get_object(
            "inc_1",
            OppConfigType.READ_ONLY,
            path=self.NEW_FILE_INC_OTHER,
        )
        self.assertEqual(opp["*inc_2_without_config"], '"AlertSender"')
        self.assertEqual(opp["valdummy"], '"dummyValue"')

    def test_is_json_list(self):
        opp = self.get_object("ObjectTypes", OppConfigType.READ_ONLY, path=self.DEFAULT_FILE)

        # Truthy
        self.assertTrue(opp._is_json_list('[1,2,3]'))
        self.assertTrue(opp._is_json_list('[1,2,3]'))
        self.assertTrue(opp._is_json_list('[{"key" : 1}, {"key" : 2}]'))
        self.assertTrue(opp._is_json_list("[]"))

        # Falsy
        self.assertFalse(opp._is_json_list("Hello World!"))
        self.assertFalse(opp._is_json_list(42))
        self.assertFalse(opp._is_json_list(""))
        self.assertFalse(opp._is_json_list('{"key_1": 1, "key_2" : [1,2,3]}'))
        self.assertFalse(opp._is_json_list('{"key": 1, child: {"key" : 2}}'))
        self.assertFalse(opp._is_json_list("{}"))

    def test_ObjectValue_eq(self):
        self.assertTrue(self.ov == self.ov)
        self.assertTrue(self.ov == ObjectValue(name='X', value='{"key_1": [1], "key_2": [42]}'))
        self.assertFalse(self.ov == ObjectValue(name='Y', value='{"key_1": [1], "key_2": [42]}'))
        self.assertFalse(self.ov == "X")

    def test_ObjectValue_from_string(self):
        self.assertTrue(ObjectValue.fromString(self.ov_str) == str(self.ov))
        self.assertFalse(ObjectValue.fromString(f'{self.ov_value}') == self.ov)

    def test_ObjectValue_str(self):
        ov_name = 'X'
        ov_value = {"key_1": [1], "key_2": [42]}
        ov = ObjectValue(name=ov_name, value=ov_value)
        self.assertEqual(str(ov), f'{ov_name}{{key_1: [1], key_2: [42]}}')

    def test_ObjectValue_as_dict(self):
        dict_result = {'key_1': [1], 'key_2': [42]}
        self.assertEqual(self.ov.as_dict, dict_result)

        # not quoted keys must work. 
        self.assertEqual(ObjectValue(name='Y', value='{no_quotes :    [42]}'), "Y{no_quotes: [42]}")
        # value must produce a valid json parsed by hjson (much mor relaxed than json itself)
        # However, it must still be a valid dict!
        self.assertRaises(ValueError, lambda : ObjectValue(name='Y', value='{no_qu; :: aef426&%8otes : [42]}'))   
        ov_as_dict = ObjectValue(name=self.ov_name, value=self.ov_value)
        self.assertTrue(ov_as_dict == self.ov)
        ov_as_dict.as_dict = {"key_1": [2]}
        self.assertFalse(ov_as_dict == self.ov)
        ov_as_dict.add(key="key_1", value=[1], override=True)
        ov_as_dict.add(key="key_2", value=[42], override=True)
        self.assertTrue(ov_as_dict == self.ov)
        ov_as_dict.add(key="key_2", value=[1], override=False)
        self.assertTrue(ov_as_dict == self.ov)
        ov_as_dict.add(key="key_2", value=[1], override=True)
        self.assertFalse(ov_as_dict == self.ov)

    def test_ObjectValue_parse_string(self):
        self.assertEqual(ObjectValue.parse_string(f'{self.ov_name}{self.ov_value}'), [self.ov_name, self.ov_value_str])
        self.assertEqual(ObjectValue.parse_string(f'{""}{self.ov_value}'), ["", self.ov_value_str])
        self.assertEqual(ObjectValue.parse_string(f'{""}{"{}"}'), ["", "{}"])

        self.assertRaises(AttributeError, lambda: ObjectValue.parse_string(f'{self.ov_name}{""}'))
        self.assertRaises(AttributeError, lambda: ObjectValue.parse_string(f'{""}{""}'))

    def test_read_object_types(self):
        result_opt_7: ObjectValue = ObjectValue(name='', value='{"key_1" : [1], "key_2" : [42]}')
        result_opt_8: List[int] = [1, 2, 3, 4]
        result_opt_9: ObjectValue = ObjectValue(name='', value='{"key_1" : {"key_1_1": 1, "key_1_2": 2}}')
        result_opt_10: List[Dict[str, int]] = [{"key_1": 1, "key_2": 2}, {"key_1": 3, "key_2": 4}]
        result_opt_11: ObjectValue = ObjectValue(name='X',
                                                 value='{"clockRate": 33, "sequencenumber": 100, "key3": "foobar"}')
        result_opt_12: ObjectValue = ObjectValue(name='X', value='{clockRate: 33, sequencenumber: 100, key3: "foobar"}')

        object_types = self.get_object("ObjectTypes", OppConfigType.READ_ONLY, path=self.DEFAULT_FILE)
        opt_7 = object_types["opt_7"]
        opt_8 = object_types["opt_8"]
        opt_9 = object_types["opt_9"]
        opt_10 = object_types["opt_10"]
        opt_11 = object_types["opt_11"]
        opt_12 = object_types["opt_12"]

        self.assertEqual(result_opt_7, opt_7)
        self.assertEqual(result_opt_8, opt_8)
        self.assertEqual(result_opt_9, opt_9)
        self.assertEqual(result_opt_10, opt_10)
        self.assertEqual(result_opt_11, opt_11)
        self.assertEqual(result_opt_12, opt_12)
        self.assertRaises(KeyError, lambda: object_types["opt_13"])


if __name__ == '__main__':
    print("pre main")
    unittest.main()
    print("post main")

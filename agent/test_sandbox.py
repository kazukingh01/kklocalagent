from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sandbox import (
    CommandNotAllowed,
    PathOutsideRoot,
    ensure_command_allowed,
    ensure_path_in_root,
    parse_shell_command,
)

ALLOW = {"date", "ip", "uname", "df", "uptime", "who"}


class ParseShellCommandTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_shell_command("date"), ["date"])

    def test_with_args(self):
        self.assertEqual(parse_shell_command("uname -a"), ["uname", "-a"])

    def test_quoted(self):
        self.assertEqual(parse_shell_command('date "+%Y"'), ["date", "+%Y"])

    def test_empty_raises(self):
        with self.assertRaises(CommandNotAllowed):
            parse_shell_command("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(CommandNotAllowed):
            parse_shell_command("   ")


class EnsureCommandAllowedTests(unittest.TestCase):
    def test_allowed(self):
        self.assertEqual(ensure_command_allowed("date", ALLOW), ["date"])
        self.assertEqual(ensure_command_allowed("uptime", ALLOW), ["uptime"])

    def test_allowed_with_args(self):
        self.assertEqual(
            ensure_command_allowed("uname -a", ALLOW),
            ["uname", "-a"],
        )

    def test_denied_simple(self):
        with self.assertRaises(CommandNotAllowed) as ctx:
            ensure_command_allowed("rm -rf /", ALLOW)
        self.assertIn("rm", str(ctx.exception))

    def test_denied_path_disguised(self):
        with self.assertRaises(CommandNotAllowed):
            ensure_command_allowed("/bin/date", ALLOW)

    def test_denied_substring_safe(self):
        allow_with_rmdir = ALLOW | {"rmdir"}
        with self.assertRaises(CommandNotAllowed):
            ensure_command_allowed("rm -rf /", allow_with_rmdir)


class EnsurePathInRootTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "sub").mkdir()
        (self.root / "sub" / "data.txt").write_text("hello")

    def tearDown(self):
        self._tmp.cleanup()

    def test_relative_ok(self):
        p = ensure_path_in_root("sub/data.txt", str(self.root))
        self.assertEqual(p, self.root / "sub" / "data.txt")

    def test_root_self(self):
        with self.assertRaises(PathOutsideRoot):
            ensure_path_in_root("", str(self.root))

    def test_absolute_outside_denied(self):
        with self.assertRaises(PathOutsideRoot):
            ensure_path_in_root("/etc/passwd", str(self.root))

    def test_dotdot_traversal_denied(self):
        with self.assertRaises(PathOutsideRoot):
            ensure_path_in_root("../etc/passwd", str(self.root))

    def test_dotdot_deep_traversal_denied(self):
        with self.assertRaises(PathOutsideRoot):
            ensure_path_in_root("sub/../../etc/passwd", str(self.root))

    def test_symlink_jailbreak_denied(self):
        outside = Path(tempfile.mkdtemp())
        try:
            (outside / "secret.txt").write_text("secret")
            (self.root / "link").symlink_to(outside / "secret.txt")
            with self.assertRaises(PathOutsideRoot):
                ensure_path_in_root("link", str(self.root))
        finally:
            (outside / "secret.txt").unlink()
            outside.rmdir()


if __name__ == "__main__":
    unittest.main()

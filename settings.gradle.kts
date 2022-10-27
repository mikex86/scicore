@file:Suppress("INACCESSIBLE_TYPE")
import org.gradle.internal.os.OperatingSystem;
import java.lang.System.getenv

rootProject.name = "scicore"
include("matplotlib")
if (OperatingSystem.current() != OperatingSystem.MAC_OS && !getenv("CI").isNullOrEmpty()) {
    include("cuda-backend")
}
include("core")
include("genericcpu-backend")
include("tests")

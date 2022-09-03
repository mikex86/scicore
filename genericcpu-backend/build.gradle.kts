import org.gradle.internal.os.OperatingSystem

plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/skija/maven")
}

val lwjglVersion = "3.3.1"

val lwjglNatives = Pair(
    System.getProperty("os.name")!!,
    System.getProperty("os.arch")!!
).let { (name, arch) ->
    when {
        arrayOf("Linux", "FreeBSD", "SunOS", "Unit").any { name.startsWith(it) } ->
            if (arrayOf("arm", "aarch64").any { arch.startsWith(it) })
                "natives-linux${if (arch.contains("64") || arch.startsWith("armv8")) "-arm64" else "-arm32"}"
            else
                "natives-linux"

        arrayOf("Mac OS X", "Darwin").any { name.startsWith(it) } ->
            "natives-macos${if (arch.startsWith("aarch64")) "-arm64" else ""}"

        arrayOf("Windows").any { name.startsWith(it) } ->
            if (arch.contains("64"))
                "natives-windows${if (arch.startsWith("aarch64")) "-arm64" else ""}"
            else
                "natives-windows-x86"

        else -> throw Error("Unrecognized or unsupported platform. Please set \"lwjglNatives\" manually")
    }
}

dependencies {
    implementation("org.jetbrains:annotations:23.0.0")

    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.9.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")

    implementation("org.apache.logging.log4j:log4j-core:2.18.0")
    implementation(project(":core"))

    // LWJGL
    implementation(platform("org.lwjgl:lwjgl-bom:$lwjglVersion"))
    implementation("org.lwjgl", "lwjgl")
    implementation("org.lwjgl", "lwjgl-jemalloc")

    // LWJGL natives
    runtimeOnly("org.lwjgl", "lwjgl", classifier = lwjglNatives)
    runtimeOnly("org.lwjgl", "lwjgl-jemalloc", classifier = lwjglNatives)
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}

tasks.create("buildNativeLib") {
    // Builds generic-cpu native library
    // builds the cmake project in ./cmake
    // and copies the resulting .so/.dll/.dylib to ./src/main/resources
    // so that it can be loaded by the JVM
    doLast {
        val cmakeBuildDir = file("./cmake/build")
        mkdir(cmakeBuildDir)

        // run cmake with ninja as generator
//        exec {
//            commandLine = listOf("cmake", "..", "-DCMAKE_BUILD_TYPE=Release")
//            workingDir = cmakeBuildDir
//        }
//
//        // build library with all threads
//        exec {
//            commandLine = listOf("cmake", "--build", ".", "--config", "Release")
//            workingDir = cmakeBuildDir
//        }

        // copy libscicore_genericcpu.dll/.so/.dylib to resources
        copy {
            @Suppress("INACCESSIBLE_TYPE")
            val libName = when (OperatingSystem.current()) {
                OperatingSystem.WINDOWS -> "**/scicore_genericcpu.dll"
                OperatingSystem.MAC_OS -> "**/libscicore_genericcpu.dylib"
                OperatingSystem.LINUX -> "**/libscicore_genericcpu.so"
                else -> throw Error("Unsupported platform")
            }
            // resolve recursively to find the file
            val libFile = fileTree(cmakeBuildDir).matching {
                include(libName)
            }.singleFile

            from(libFile)

            mkdir("src/main/resources")
            into("src/main/resources")
        }
    }
}

// java build depends on native build
tasks.getByName<JavaCompile>("compileJava") {
    dependsOn("buildNativeLib")
}
import org.gradle.internal.os.OperatingSystem
import java.lang.System.getenv
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("java")
    kotlin("jvm") version "1.7.21"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/skija/maven")
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.9.0")
    testImplementation("org.junit.platform:junit-platform-suite-api:1.9.0")
    testImplementation("org.junit.jupiter:junit-jupiter-engine")

    testImplementation("me.tongfei:progressbar:0.9.5")

    implementation("org.jetbrains:annotations:23.0.0")
    implementation(project(":matplotlib"))
    implementation(project(":core"))
    implementation(project(":genericcpu-backend"))
    @Suppress("INACCESSIBLE_TYPE")
    if (OperatingSystem.current() != OperatingSystem.MAC_OS && getenv("CI").isNullOrEmpty()) {
        implementation(project(":cuda-backend"))
    }
    implementation(kotlin("stdlib-jdk8"))
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()

    testLogging {
        events("passed", "skipped", "failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        showExceptions = true
        showCauses = true
        showStackTraces = true
    }
}
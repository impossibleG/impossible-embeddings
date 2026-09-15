# Security Policy

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Use GitHub's private vulnerability reporting for
this repository. Include affected versions, impact, and minimal reproduction steps. Reports will
be acknowledged when the project has an active maintainer available; no response-time guarantee
is made before the first stable release.

## Security defaults

The planned server binds to loopback by default, never enables remote access implicitly, and
never logs inference inputs. Network exposure will require explicit configuration and an
authentication policy. Model artifacts will be verified before activation.

The current foundation contains no listening network service.

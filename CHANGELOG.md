# Changelog

All notable changes to Video Scanner will be documented in this file.

## [v1.0.0] - 2024-03-28
### Added
- Initial release
- GPU-accelerated video analysis
- Frame drop detection
- Stuttering analysis
- FPS monitoring
- Audio glitch detection
- Audio loss detection
- Drag & drop support
- Cache system for faster re-analysis

### Changed
- None (initial release)

### Fixed
- None (initial release)

## Version Tag Format

Tags follow [Semantic Versioning](https://semver.org/):
- `v{major}.{minor}.{patch}`
- Example: `v1.0.0`

### Tag Categories
- `major`: Breaking changes
- `minor`: New features
- `patch`: Bug fixes

### Create a New Release
```bash
# Create tag with message
git tag -a v1.0.0 -m "Initial release with GPU support"

# Push tag
git push origin v1.0.0
```

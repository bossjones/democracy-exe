"""
This type stub file was generated by pyright.
"""

from stone.backends.python_rsrc import stone_base as bb

class CameraUploadsPolicyState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.CameraUploadsPolicyState.disabled: Background camera
        uploads are disabled.
    :ivar team_policies.CameraUploadsPolicyState.enabled: Background camera
        uploads are allowed.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


CameraUploadsPolicyState_validator = ...
class ComputerBackupPolicyState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.ComputerBackupPolicyState.disabled: Computer Backup
        feature is disabled.
    :ivar team_policies.ComputerBackupPolicyState.enabled: Computer Backup
        feature is enabled.
    :ivar team_policies.ComputerBackupPolicyState.default: Computer Backup
        defaults to ON for SSB teams, and OFF for Enterprise teams.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    default = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_default(self):
        """
        Check if the union tag is ``default``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


ComputerBackupPolicyState_validator = ...
class EmmState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.EmmState.disabled: Emm token is disabled.
    :ivar team_policies.EmmState.optional: Emm token is optional.
    :ivar team_policies.EmmState.required: Emm token is required.
    """
    _catch_all = ...
    disabled = ...
    optional = ...
    required = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_optional(self):
        """
        Check if the union tag is ``optional``.

        :rtype: bool
        """
        ...
    
    def is_required(self):
        """
        Check if the union tag is ``required``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


EmmState_validator = ...
class ExternalDriveBackupPolicyState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.ExternalDriveBackupPolicyState.disabled: External Drive
        Backup feature is disabled.
    :ivar team_policies.ExternalDriveBackupPolicyState.enabled: External Drive
        Backup feature is enabled.
    :ivar team_policies.ExternalDriveBackupPolicyState.default: External Drive
        Backup default value based on team tier.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    default = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_default(self):
        """
        Check if the union tag is ``default``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


ExternalDriveBackupPolicyState_validator = ...
class FileLockingPolicyState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.FileLockingPolicyState.disabled: File locking feature is
        disabled.
    :ivar team_policies.FileLockingPolicyState.enabled: File locking feature is
        allowed.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


FileLockingPolicyState_validator = ...
class FileProviderMigrationPolicyState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.FileProviderMigrationPolicyState.disabled: Team admin
        has opted out of File Provider Migration for team members.
    :ivar team_policies.FileProviderMigrationPolicyState.enabled: Team admin has
        not opted out of File Provider Migration for team members.
    :ivar team_policies.FileProviderMigrationPolicyState.default: Team admin has
        default value based on team tier.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    default = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_default(self):
        """
        Check if the union tag is ``default``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


FileProviderMigrationPolicyState_validator = ...
class GroupCreation(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.GroupCreation.admins_and_members: Team admins and
        members can create groups.
    :ivar team_policies.GroupCreation.admins_only: Only team admins can create
        groups.
    """
    _catch_all = ...
    admins_and_members = ...
    admins_only = ...
    def is_admins_and_members(self):
        """
        Check if the union tag is ``admins_and_members``.

        :rtype: bool
        """
        ...
    
    def is_admins_only(self):
        """
        Check if the union tag is ``admins_only``.

        :rtype: bool
        """
        ...
    


GroupCreation_validator = ...
class OfficeAddInPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.OfficeAddInPolicy.disabled: Office Add-In is disabled.
    :ivar team_policies.OfficeAddInPolicy.enabled: Office Add-In is enabled.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


OfficeAddInPolicy_validator = ...
class PaperDefaultFolderPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.PaperDefaultFolderPolicy.everyone_in_team: Everyone in
        team will be the default option when creating a folder in Paper.
    :ivar team_policies.PaperDefaultFolderPolicy.invite_only: Invite only will
        be the default option when creating a folder in Paper.
    """
    _catch_all = ...
    everyone_in_team = ...
    invite_only = ...
    other = ...
    def is_everyone_in_team(self):
        """
        Check if the union tag is ``everyone_in_team``.

        :rtype: bool
        """
        ...
    
    def is_invite_only(self):
        """
        Check if the union tag is ``invite_only``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PaperDefaultFolderPolicy_validator = ...
class PaperDeploymentPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.PaperDeploymentPolicy.full: All team members have access
        to Paper.
    :ivar team_policies.PaperDeploymentPolicy.partial: Only whitelisted team
        members can access Paper. To see which user is whitelisted, check
        'is_paper_whitelisted' on 'account/info'.
    """
    _catch_all = ...
    full = ...
    partial = ...
    other = ...
    def is_full(self):
        """
        Check if the union tag is ``full``.

        :rtype: bool
        """
        ...
    
    def is_partial(self):
        """
        Check if the union tag is ``partial``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PaperDeploymentPolicy_validator = ...
class PaperDesktopPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.PaperDesktopPolicy.disabled: Do not allow team members
        to use Paper Desktop.
    :ivar team_policies.PaperDesktopPolicy.enabled: Allow team members to use
        Paper Desktop.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PaperDesktopPolicy_validator = ...
class PaperEnabledPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.PaperEnabledPolicy.disabled: Paper is disabled.
    :ivar team_policies.PaperEnabledPolicy.enabled: Paper is enabled.
    :ivar team_policies.PaperEnabledPolicy.unspecified: Unspecified policy.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    unspecified = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_unspecified(self):
        """
        Check if the union tag is ``unspecified``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PaperEnabledPolicy_validator = ...
class PasswordControlMode(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.PasswordControlMode.disabled: Password is disabled.
    :ivar team_policies.PasswordControlMode.enabled: Password is enabled.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PasswordControlMode_validator = ...
class PasswordStrengthPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.PasswordStrengthPolicy.minimal_requirements: User
        passwords will adhere to the minimal password strength policy.
    :ivar team_policies.PasswordStrengthPolicy.moderate_password: User passwords
        will adhere to the moderate password strength policy.
    :ivar team_policies.PasswordStrengthPolicy.strong_password: User passwords
        will adhere to the very strong password strength policy.
    """
    _catch_all = ...
    minimal_requirements = ...
    moderate_password = ...
    strong_password = ...
    other = ...
    def is_minimal_requirements(self):
        """
        Check if the union tag is ``minimal_requirements``.

        :rtype: bool
        """
        ...
    
    def is_moderate_password(self):
        """
        Check if the union tag is ``moderate_password``.

        :rtype: bool
        """
        ...
    
    def is_strong_password(self):
        """
        Check if the union tag is ``strong_password``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PasswordStrengthPolicy_validator = ...
class RolloutMethod(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.RolloutMethod.unlink_all: Unlink all.
    :ivar team_policies.RolloutMethod.unlink_most_inactive: Unlink devices with
        the most inactivity.
    :ivar team_policies.RolloutMethod.add_member_to_exceptions: Add member to
        Exceptions.
    """
    _catch_all = ...
    unlink_all = ...
    unlink_most_inactive = ...
    add_member_to_exceptions = ...
    def is_unlink_all(self):
        """
        Check if the union tag is ``unlink_all``.

        :rtype: bool
        """
        ...
    
    def is_unlink_most_inactive(self):
        """
        Check if the union tag is ``unlink_most_inactive``.

        :rtype: bool
        """
        ...
    
    def is_add_member_to_exceptions(self):
        """
        Check if the union tag is ``add_member_to_exceptions``.

        :rtype: bool
        """
        ...
    


RolloutMethod_validator = ...
class SharedFolderBlanketLinkRestrictionPolicy(bb.Union):
    """
    Policy governing whether shared folder membership is required to access
    shared links.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SharedFolderBlanketLinkRestrictionPolicy.members: Only
        members of shared folders can access folder content via shared link.
    :ivar team_policies.SharedFolderBlanketLinkRestrictionPolicy.anyone: Anyone
        can access folder content via shared link.
    """
    _catch_all = ...
    members = ...
    anyone = ...
    other = ...
    def is_members(self):
        """
        Check if the union tag is ``members``.

        :rtype: bool
        """
        ...
    
    def is_anyone(self):
        """
        Check if the union tag is ``anyone``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SharedFolderBlanketLinkRestrictionPolicy_validator = ...
class SharedFolderJoinPolicy(bb.Union):
    """
    Policy governing which shared folders a team member can join.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SharedFolderJoinPolicy.from_team_only: Team members can
        only join folders shared by teammates.
    :ivar team_policies.SharedFolderJoinPolicy.from_anyone: Team members can
        join any shared folder, including those shared by users outside the
        team.
    """
    _catch_all = ...
    from_team_only = ...
    from_anyone = ...
    other = ...
    def is_from_team_only(self):
        """
        Check if the union tag is ``from_team_only``.

        :rtype: bool
        """
        ...
    
    def is_from_anyone(self):
        """
        Check if the union tag is ``from_anyone``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SharedFolderJoinPolicy_validator = ...
class SharedFolderMemberPolicy(bb.Union):
    """
    Policy governing who can be a member of a folder shared by a team member.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SharedFolderMemberPolicy.team: Only a teammate can be a
        member of a folder shared by a team member.
    :ivar team_policies.SharedFolderMemberPolicy.anyone: Anyone can be a member
        of a folder shared by a team member.
    """
    _catch_all = ...
    team = ...
    anyone = ...
    other = ...
    def is_team(self):
        """
        Check if the union tag is ``team``.

        :rtype: bool
        """
        ...
    
    def is_anyone(self):
        """
        Check if the union tag is ``anyone``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SharedFolderMemberPolicy_validator = ...
class SharedLinkCreatePolicy(bb.Union):
    """
    Policy governing the visibility of shared links. This policy can apply to
    newly created shared links, or all shared links.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SharedLinkCreatePolicy.default_public: By default,
        anyone can access newly created shared links. No login will be required
        to access the shared links unless overridden.
    :ivar team_policies.SharedLinkCreatePolicy.default_team_only: By default,
        only members of the same team can access newly created shared links.
        Login will be required to access the shared links unless overridden.
    :ivar team_policies.SharedLinkCreatePolicy.team_only: Only members of the
        same team can access all shared links. Login will be required to access
        all shared links.
    :ivar team_policies.SharedLinkCreatePolicy.default_no_one: Only people
        invited can access newly created links. Login will be required to access
        the shared links unless overridden.
    """
    _catch_all = ...
    default_public = ...
    default_team_only = ...
    team_only = ...
    default_no_one = ...
    other = ...
    def is_default_public(self):
        """
        Check if the union tag is ``default_public``.

        :rtype: bool
        """
        ...
    
    def is_default_team_only(self):
        """
        Check if the union tag is ``default_team_only``.

        :rtype: bool
        """
        ...
    
    def is_team_only(self):
        """
        Check if the union tag is ``team_only``.

        :rtype: bool
        """
        ...
    
    def is_default_no_one(self):
        """
        Check if the union tag is ``default_no_one``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SharedLinkCreatePolicy_validator = ...
class ShowcaseDownloadPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.ShowcaseDownloadPolicy.disabled: Do not allow files to
        be downloaded from Showcases.
    :ivar team_policies.ShowcaseDownloadPolicy.enabled: Allow files to be
        downloaded from Showcases.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


ShowcaseDownloadPolicy_validator = ...
class ShowcaseEnabledPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.ShowcaseEnabledPolicy.disabled: Showcase is disabled.
    :ivar team_policies.ShowcaseEnabledPolicy.enabled: Showcase is enabled.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


ShowcaseEnabledPolicy_validator = ...
class ShowcaseExternalSharingPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.ShowcaseExternalSharingPolicy.disabled: Do not allow
        showcases to be shared with people not on the team.
    :ivar team_policies.ShowcaseExternalSharingPolicy.enabled: Allow showcases
        to be shared with people not on the team.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


ShowcaseExternalSharingPolicy_validator = ...
class SmartSyncPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SmartSyncPolicy.local: The specified content will be
        synced as local files by default.
    :ivar team_policies.SmartSyncPolicy.on_demand: The specified content will be
        synced as on-demand files by default.
    """
    _catch_all = ...
    local = ...
    on_demand = ...
    other = ...
    def is_local(self):
        """
        Check if the union tag is ``local``.

        :rtype: bool
        """
        ...
    
    def is_on_demand(self):
        """
        Check if the union tag is ``on_demand``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SmartSyncPolicy_validator = ...
class SmarterSmartSyncPolicyState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SmarterSmartSyncPolicyState.disabled: Smarter Smart Sync
        feature is disabled.
    :ivar team_policies.SmarterSmartSyncPolicyState.enabled: Smarter Smart Sync
        feature is enabled.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SmarterSmartSyncPolicyState_validator = ...
class SsoPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SsoPolicy.disabled: Users will be able to sign in with
        their Dropbox credentials.
    :ivar team_policies.SsoPolicy.optional: Users will be able to sign in with
        either their Dropbox or single sign-on credentials.
    :ivar team_policies.SsoPolicy.required: Users will be required to sign in
        with their single sign-on credentials.
    """
    _catch_all = ...
    disabled = ...
    optional = ...
    required = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_optional(self):
        """
        Check if the union tag is ``optional``.

        :rtype: bool
        """
        ...
    
    def is_required(self):
        """
        Check if the union tag is ``required``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SsoPolicy_validator = ...
class SuggestMembersPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.SuggestMembersPolicy.disabled: Suggest members is
        disabled.
    :ivar team_policies.SuggestMembersPolicy.enabled: Suggest members is
        enabled.
    """
    _catch_all = ...
    disabled = ...
    enabled = ...
    other = ...
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_enabled(self):
        """
        Check if the union tag is ``enabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SuggestMembersPolicy_validator = ...
class TeamMemberPolicies(bb.Struct):
    """
    Policies governing team members.

    :ivar team_policies.TeamMemberPolicies.sharing: Policies governing sharing.
    :ivar team_policies.TeamMemberPolicies.emm_state: This describes the
        Enterprise Mobility Management (EMM) state for this team. This
        information can be used to understand if an organization is integrating
        with a third-party EMM vendor to further manage and apply restrictions
        upon the team's Dropbox usage on mobile devices. This is a new feature
        and in the future we'll be adding more new fields and additional
        documentation.
    :ivar team_policies.TeamMemberPolicies.office_addin: The admin policy around
        the Dropbox Office Add-In for this team.
    :ivar team_policies.TeamMemberPolicies.suggest_members_policy: The team
        policy on if teammembers are allowed to suggest users for admins to
        invite to the team.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, sharing=..., emm_state=..., office_addin=..., suggest_members_policy=...) -> None:
        ...
    
    sharing = ...
    emm_state = ...
    office_addin = ...
    suggest_members_policy = ...


TeamMemberPolicies_validator = ...
class TeamSharingPolicies(bb.Struct):
    """
    Policies governing sharing within and outside of the team.

    :ivar team_policies.TeamSharingPolicies.shared_folder_member_policy: Who can
        join folders shared by team members.
    :ivar team_policies.TeamSharingPolicies.shared_folder_join_policy: Which
        shared folders team members can join.
    :ivar team_policies.TeamSharingPolicies.shared_link_create_policy: Who can
        view shared links owned by team members.
    :ivar team_policies.TeamSharingPolicies.group_creation_policy: Who can
        create groups.
    :ivar
        team_policies.TeamSharingPolicies.shared_folder_link_restriction_policy:
        Who can view links to content in shared folders.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, shared_folder_member_policy=..., shared_folder_join_policy=..., shared_link_create_policy=..., group_creation_policy=..., shared_folder_link_restriction_policy=...) -> None:
        ...
    
    shared_folder_member_policy = ...
    shared_folder_join_policy = ...
    shared_link_create_policy = ...
    group_creation_policy = ...
    shared_folder_link_restriction_policy = ...


TeamSharingPolicies_validator = ...
class TwoStepVerificationPolicy(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.TwoStepVerificationPolicy.require_tfa_enable: Enabled
        require two factor authorization.
    :ivar team_policies.TwoStepVerificationPolicy.require_tfa_disable: Disabled
        require two factor authorization.
    """
    _catch_all = ...
    require_tfa_enable = ...
    require_tfa_disable = ...
    other = ...
    def is_require_tfa_enable(self):
        """
        Check if the union tag is ``require_tfa_enable``.

        :rtype: bool
        """
        ...
    
    def is_require_tfa_disable(self):
        """
        Check if the union tag is ``require_tfa_disable``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


TwoStepVerificationPolicy_validator = ...
class TwoStepVerificationState(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar team_policies.TwoStepVerificationState.required: Enabled require two
        factor authorization.
    :ivar team_policies.TwoStepVerificationState.optional: Optional require two
        factor authorization.
    :ivar team_policies.TwoStepVerificationState.disabled: Disabled require two
        factor authorization.
    """
    _catch_all = ...
    required = ...
    optional = ...
    disabled = ...
    other = ...
    def is_required(self):
        """
        Check if the union tag is ``required``.

        :rtype: bool
        """
        ...
    
    def is_optional(self):
        """
        Check if the union tag is ``optional``.

        :rtype: bool
        """
        ...
    
    def is_disabled(self):
        """
        Check if the union tag is ``disabled``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


TwoStepVerificationState_validator = ...
ROUTES = ...
module Utils (
    ShowBoolKind (..),
    showBool,
) where
    class ShowBoolKind (b :: Bool) where
        showBool :: String

    instance ShowBoolKind 'True where
        showBool = "True"

    instance ShowBoolKind 'False where
        showBool = "False"
